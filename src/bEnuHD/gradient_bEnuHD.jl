# function sumexp(vec::Array{Float64,1})
#     mysum = 0.0
#     @inbounds @simd for i=1:length(vec)
#         mysum += exp(vec[i])
#     end
#     return mysum
# end

function logsumexp(X::AbstractArray{T}) where {T<:Real}
    isempty(X) && return log(zero(T))
    u = maximum(X)
    isfinite(u) || return float(u)
    let u=u # avoid https://github.com/JuliaLang/julia/issues/15276
        u + log(sum(x -> exp(x-u), X))
    end
end

""" Pseudo-Likelihood gradient

	H = β * E - ν*∑δ(σᵢ,σᵢᵂᵀ) + η∏δ(σᵢ,σᵢᵂᵀ)

"""


function PLsiteAndGrad!(x::Vector{Float64}, grad::Vector{Float64}, w::Workspace, site::Int,
	 beta::Vector{Float64})


    LL = w.LL
    q = w.q
	q2 = q*q
    N = w.N
	T = w.T
	wt = w.wt_seq

	@assert T == length(beta)

    for i=1:LL-q
		grad[i] = 2.0 * w.reg.lambdaJE * x[i]
		#grad[i] = 0.0
	end

	for i=(LL-q+1):LL
		grad[i] = 4.0 * w.reg.lambdaHE * x[i]
		#grad[i] = 0.0
	end

	grad[LL+1:end] .= 0.0

	pseudolike = 0.0
	vecene = zeros(Float64,q)
	expvecenesumnorm = zeros(Float64,q)
	lnorm = 0.0

	for t = 1:T

        M = w.samples[t].M
    	W = sdata(w.samples[t].W)
		Z = sdata(w.samples[t].Z)
		IdxZ = sdata(w.samples[t].IdxZ)
		idxRound = union(1:LL,LL+t)
		b = beta[t]

	   @inbounds for a = 1:M

		    fillvecene!(vecene, x[idxRound], site, view(IdxZ,:,a), b, q, N, wt[site])

			lnorm = logsumexp(vecene)
	        expvecenesumnorm .= @. exp(vecene - lnorm)

			pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)

			#del_prod = 1

	        @simd for i = 1:site-1
	             @simd for s = 1:q
				 	grad[ IdxZ[i,a] + s  ] += W[a] * b * expvecenesumnorm[s]

				 end

				grad[ IdxZ[i,a] + Z[site,a] ] -= W[a] * b
				#del_prod *= (Z[i,a]==wt[i])
	        end
            @simd for i = site+1:N
	            @simd for s = 1:q
					grad[ IdxZ[i,a] - q2 + s ] += W[a] * b * expvecenesumnorm[s]

	            end

				grad[ IdxZ[i,a] - q2 + Z[site,a] ] -= W[a] * b
				#del_prod *= (Z[i,a]==wt[i])
	        end
	        @simd for s = 1:q
	            grad[ (N-1) * q2 + s ] += W[a] * b * expvecenesumnorm[s]

	        end

			grad[ (N-1) * q2 + Z[site,a] ] -= W[a] * b
			grad[ (N-1) * q2 + wt[site] ] += W[a]*4/T*w.reg.lambdaHE*x[LL+t]
			grad[LL+t] += W[a]*(-(Z[site,a]==wt[site]) + expvecenesumnorm[wt[site]]+4/T*w.reg.lambdaHE*(x[LL+t]+x[(N-1)*q2+wt[site]]))
			#grad[LL+t+T] += W[a]*del_prod*( (Z[site,a]==wt[site]) - expvecenesumnorm[wt[site]] )
		end
		#println(grad[LL+t])
		#println(grad[LL+t+T])
	end


	 #pseudolike += L2norm(x[1:LL], w)
	 pseudolike += L2normFields(x, site, w)
    return pseudolike
end

#TRIAL WITH NU INDEPENDENT REGULARIZATION

function PLsiteAndGrad!(x::Vector{Float64}, grad::Vector{Float64}, w::Workspace, site::Int,
	beta::Vector{Float64}, lNu::Float64)


   LL = w.LL
   q = w.q
   q2 = q*q
   N = w.N
   T = w.T
   wt = w.wt_seq

   @assert T == length(beta)

   for i=1:LL-q
	   grad[i] = 2.0 * w.reg.lambdaJE * x[i]
	   #grad[i] = 0.0
   end

   for i=(LL-q+1):LL
	   grad[i] = 4.0 * w.reg.lambdaHE * x[i]
	   #grad[i] = 0.0
   end

   for t=1:T
	grad[LL+t] = 2.0/T*lNu*x[LL+t]
   end
   

   pseudolike = 0.0
   vecene = zeros(Float64,q)
   expvecenesumnorm = zeros(Float64,q)
   lnorm = 0.0

   for t = 1:T

	   M = w.samples[t].M
	   W = sdata(w.samples[t].W)
	   Z = sdata(w.samples[t].Z)
	   IdxZ = sdata(w.samples[t].IdxZ)
	   idxRound = union(1:LL,LL+t)
	   b = beta[t]

	  @inbounds for a = 1:M

		   fillvecene!(vecene, x[idxRound], site, view(IdxZ,:,a), b, q, N, wt[site])

		   lnorm = logsumexp(vecene)
		   expvecenesumnorm .= @. exp(vecene - lnorm)

		   pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)

		   @simd for i = 1:site-1
				@simd for s = 1:q
					grad[ IdxZ[i,a] + s  ] += W[a] * b * expvecenesumnorm[s]
					#grad[ IdxZ[i,a] + s  ] = 0.0
				end

			   grad[ IdxZ[i,a] + Z[site,a] ] -= W[a] * b
			   #grad[ IdxZ[i,a] + Z[site,a] ] = 0.0
		   end
		   @simd for i = site+1:N
			   @simd for s = 1:q
				   grad[ IdxZ[i,a] - q2 + s ] += W[a] * b * expvecenesumnorm[s]
				   #grad[ IdxZ[i,a] - q2 + s ] = 0.0
			   end

			   grad[ IdxZ[i,a] - q2 + Z[site,a] ] -= W[a] * b
			   #grad[ IdxZ[i,a] - q2 + Z[site,a] ] = 0.0
		   end
		   @simd for s = 1:q
			   grad[ (N-1) * q2 + s ] += W[a] * b * expvecenesumnorm[s]
			   #grad[ (N-1) * q2 + s ] = 0.0
		   end

		   grad[ (N-1) * q2 + Z[site,a] ] -= W[a] * b
		   #grad[ (N-1) * q2 + Z[site,a] ] = 0.0
		   grad[LL+t] += W[a]*(-(Z[site,a]==wt[site]) + expvecenesumnorm[wt[site]])
	   end
   end


	pseudolike += L2normNu(x, w, lNu)
   return pseudolike
end

#Nu set of values furnished and not learned

function PLsiteAndGrad!(x::Vector{Float64}, grad::Vector{Float64}, w::Workspace, site::Int,
	beta::Vector{Float64}, nu::Vector{Float64})


   LL = w.LL
   q = w.q
   q2 = q*q
   N = w.N
   T = w.T
   wt = w.wt_seq

   @assert T == length(beta) == length(nu)

   for i=1:LL-q
	   grad[i] = 2.0 * w.reg.lambdaJE * x[i]
	   #grad[i] = 0.0
   end

   for i=(LL-q+1):LL
	   grad[i] = 4.0 * w.reg.lambdaHE * x[i]
	   #grad[i] = 0.0
   end
   

   pseudolike = 0.0
   vecene = zeros(Float64,q)
   expvecenesumnorm = zeros(Float64,q)
   lnorm = 0.0

   for t = 1:T

	   M = w.samples[t].M
	   W = sdata(w.samples[t].W)
	   Z = sdata(w.samples[t].Z)
	   IdxZ = sdata(w.samples[t].IdxZ)
	   b = beta[t]
	   ν = nu[t]

	  @inbounds for a = 1:M

		   fillvecene!(vecene, x, site, view(IdxZ,:,a), b, ν, q, N, wt[site])

		   lnorm = logsumexp(vecene)
		   expvecenesumnorm .= @. exp(vecene - lnorm)

		   pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)

		   @simd for i = 1:site-1
				@simd for s = 1:q
					grad[ IdxZ[i,a] + s  ] += W[a] * b * expvecenesumnorm[s]
				end

			   grad[ IdxZ[i,a] + Z[site,a] ] -= W[a] * b
		   end
		   @simd for i = site+1:N
			   @simd for s = 1:q
				   grad[ IdxZ[i,a] - q2 + s ] += W[a] * b * expvecenesumnorm[s]
			   end

			   grad[ IdxZ[i,a] - q2 + Z[site,a] ] -= W[a] * b
		   end
		   @simd for s = 1:q
			   grad[ (N-1) * q2 + s ] += W[a] * b * expvecenesumnorm[s]
		   end

		   grad[ (N-1) * q2 + Z[site,a] ] -= W[a] * b

	   end
   end


	pseudolike += L2norm(x[1:LL], w)
   return pseudolike
end

#######################
#Energy filling

function fillvecene!(vecene::Vector{Float64}, x::Vector{Float64}, site::Int, IdxSeq::AbstractArray{Int,1},  b::Float64, q::Int, N::Int, wt::Int)

	q2=q*q
	@assert length(x) == (N-1)*q2+q+1
    @inbounds for l = 1:q
        scra::Float64 = 0.0
        @simd for i = 1:site-1 # Begin sum_i \neq site J
			scra += b*x[IdxSeq[i] + l]
        end
        #skipping sum over residue site
    	@simd for i = site+1:N
			scra +=  b*x[IdxSeq[i] - q2 + l]
        end # End sum_i \neq site J
		scra +=  b*x[(N-1)*q2 + l] + x[end]*(l==wt[site]) # sum H and h.d.
        vecene[l] = scra
    end

end

#Nu furnished and not learned
function fillvecene!(vecene::Vector{Float64}, x::Vector{Float64}, site::Int, IdxSeq::AbstractArray{Int,1}, b::Float64, ν::Float64, q::Int, N::Int, wt::Int)

	q2=q*q
	@assert length(x) == (N-1)*q2+q
    @inbounds for l = 1:q
        scra::Float64 = 0.0
        @simd for i = 1:site-1 # Begin sum_i \neq site J
			scra += b*x[IdxSeq[i] + l]
        end
        #skipping sum over residue site
    	@simd for i = site+1:N
			scra +=  b*x[IdxSeq[i] - q2 + l]
        end # End sum_i \neq site J
		scra +=  b*x[(N-1)*q2 + l] + ν*(l==wt[site]) # sum H and h.d.
        vecene[l] = scra
    end

end


# function fillvecene!(bvecener::Vector{Float64},vecener::Vector{Float64},x::Matrix{Float64},site::Int,seq::AbstractArray{Int,1},b::Float64,q::Int,N::Int)
#
# 	q2 = q*q
#
#     @inbounds for l = 1:q
# 		offset::Int=0
#         bener::Float64 = 0.0
#         ener::Float64 = 0.0
#
#         for i = 1:site-1
#             bener += b * x[offset + l + q * (seq[i]-1), site] + x[LL + offset + l + q * (seq[i]-1), site]
#             ener += x[offset + l + q * (seq[i]-1), site]
# 			offset += q2
#         end
#
#         for i = site+1:N
# 			bener += b * x[offset + l + q * (seq[i]-1), site] + x0[offset + l + q * (seq[i]-1), site]
# 			ener += x[offset + l + q * (seq[i]-1), site]
# 			offset += q2
#         end
#
#         bener += b * x[offset + l,site] + x[LL + offset + l,site]
#         ener += x[offset + l,site]
#
#         bvecener[l] = bener
#         vecener[l] = ener
#     end
#
# end

######################
#Regularization

function L2norm(x::Array{Float64,1}, w::Workspace)
    q = w.q
    N = w.N
	LL = w.LL

	@assert LL == Int(length(x))

    mysum1E = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += x[i] * x[i]
    end
    mysum1E *= w.reg.lambdaJE

    mysum2E = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += x[i] * x[i]
    end
    mysum2E *= 2*w.reg.lambdaHE
    return mysum1E + mysum2E
end

function L2normFields(x::Array{Float64,1}, site::Int, w::Workspace)

    q = w.q
    N = w.N
	LL = w.LL
	T = w.T
	wt = w.wt_seq

	@assert LL + T == Int(length(x))

    mysum1E = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += x[i] * x[i]
    end
    mysum1E *= w.reg.lambdaJE

    mysum2E = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += x[i] * x[i]
    end
	mysum2E *= 2*w.reg.lambdaHE
	
	mysumEnu = 0.0
	for t=1:T
		mysumEnu += x[LL+t]*x[LL+t]+2*x[LL+t]*x[LL-q+wt[site]]
	end
	mysumEnu *= 2/T*w.reg.lambdaHE

    return mysum1E + mysum2E + mysumEnu
end

function L2normNu(x::Array{Float64,1}, w::Workspace, lNu::Float64)

    q = w.q
    N = w.N
	LL = w.LL
	T = w.T

	@assert LL + T == Int(length(x))

    mysum1E = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += x[i] * x[i]
    end
    mysum1E *= w.reg.lambdaJE

    mysum2E = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += x[i] * x[i]
    end
	mysum2E *= 2*w.reg.lambdaHE
	
	mysumNu = 0.0
	for t=1:T
		mysumNu += x[LL+t]*x[LL+t]
	end
	mysumNu *= lNu

    return mysum1E + mysum2E + mysumNu/T
end
##############################################################
# BETA OPTIMIZATION

# function Beta2Grad(b::Float64, x::Matrix{Float64}, Z::AbstractArray{Int,2}, W::AbstractVector{Float64}, q::Int, N::Int)
#
# 	M = length(W)
#
# 	bvecener = zeros(Float64,q)
# 	vecener = zeros(Float64,q)
# 	g = 0.0
# 	g2 = 0.0
#
# 	for m = 1:M
# 		seq = view(Z,:,m)
# 		for site = 1:N
# 			fillvecene!(bvecener, vecener, x, site, seq, b, q, N)
# 			lnorm = logsumexp(bvecener)
# 			avener = sum( @. vecener * exp(bvecener - lnorm))
# 			av2ener = sum( @. (vecener * vecener) * exp.(bvecener - lnorm))
# 			g -= W[m]*( vecener[seq[site]] - avener )
# 			g2 += W[m]*(av2ener - avener*avener)
# 		end
# 	end
#
# 	return g,g2
#
# end
