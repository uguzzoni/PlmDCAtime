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

	H = G + β * E

"""


function PLsiteAndGrad!(x::Vector{Float64}, grad::Vector{Float64}, w::Workspace, site::Int,
	 beta::Vector{Float64})


    LL = w.LL
    q = w.q
	q2 = q*q
    N = w.N
	T = w.T

	@assert T == length(beta)

    for i=1:LL-q
	    grad[i] = 2.0 * w.reg.lambdaJE * x[i]
	end

	for i=(LL-q+1):LL
	    grad[i] = 4.0 * w.reg.lambdaHE * x[i]
	end

	for i=LL+1:2*LL-q
		grad[i] = 2.0 * w.reg.lambdaJG * x[i]
		#grad[i] = 0.0
	end
	for i=(2*LL-q+1):2*LL
		grad[i] = 4.0 * w.reg.lambdaHG * x[i]
		#grad[i] = 0.0
	end

	pseudolike = 0.0
	vecene = zeros(Float64,q)
	expvecenesumnorm = zeros(Float64,q)
	lnorm = 0.0
	xx = Array{Float64}(undef,LL);

	for t = 1:T

        M = w.samples[t].M
    	W = sdata(w.samples[t].W)
		Z = sdata(w.samples[t].Z)
		IdxZ = sdata(w.samples[t].IdxZ)

		b = beta[t]

		xx .= @. b * x[1:LL] + x[LL+1:2*LL]

	   @inbounds for a = 1:M

		    fillvecene!(vecene, xx, site, view(IdxZ,:,a), q, N)

			lnorm = logsumexp(vecene)
	        expvecenesumnorm .= @. exp(vecene - lnorm)

			pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)

	        @simd for i = 1:site-1
	            @simd for s = 1:q
					grad[ IdxZ[i,a] + s  ] += W[a] * b * expvecenesumnorm[s]
					grad[ LL + IdxZ[i,a] + s ] += W[a] * expvecenesumnorm[s]
					#grad[ LL + IdxZ[i,a] + s ] = 0.0

				end

				grad[ IdxZ[i,a] + Z[site,a] ] -= W[a] * b
				grad[ LL +  IdxZ[i,a] + Z[site,a] ] -= W[a]
				#grad[ LL +  IdxZ[i,a] + Z[site,a] ] = 0.0

	        end
            @simd for i = site+1:N
	            @simd for s = 1:q
					grad[ IdxZ[i,a] - q2 + s ] += W[a] * b * expvecenesumnorm[s]
					grad[ LL + IdxZ[i,a] - q2 + s ] += W[a] * expvecenesumnorm[s]
					#grad[ LL + IdxZ[i,a] - q2 + s ] = 0.0

	            end

				grad[ IdxZ[i,a] - q2 + Z[site,a] ] -= W[a] * b
				grad[ LL + IdxZ[i,a] - q2 + Z[site,a] ] -= W[a]
				#grad[ LL + IdxZ[i,a] - q2 + Z[site,a] ] = 0.0

	        end
	        @simd for s = 1:q
	            grad[ (N-1) * q2 + s ] += W[a] * b * expvecenesumnorm[s]
				grad[ LL + (N-1) * q2 + s ] += W[a] * expvecenesumnorm[s]
				#grad[ LL + (N-1) * q2 + s ] = 0.0

	        end

			grad[ (N-1) * q2 + Z[site,a] ] -= W[a] * b
			grad[ LL + (N-1) * q2 + Z[site,a] ] -= W[a]
			#grad[ LL + (N-1) * q2 + Z[site,a] ] = 0.0

	    end
	end


 	pseudolike += L2norm(x, w)
    return pseudolike
end

#######################
#Energy filling

function fillvecene!(vecene::Vector{Float64}, x::Vector{Float64}, site::Int, IdxSeq::AbstractArray{Int,1}, q::Int, N::Int)

	q2=q*q
    @inbounds for l = 1:q
        scra::Float64 = 0.0
        @simd for i = 1:site-1 # Begin sum_i \neq site J
            scra += x[IdxSeq[i] + l]
        end
        # skipping sum over residue site
    	@simd for i = site+1:N
            scra +=  x[IdxSeq[i] - q2 + l]
        end # End sum_i \neq site J
        scra +=  x[(N-1)*q2 + l] # sum H
        vecene[l] = scra
    end

end

######################
#Regularization

function L2norm(x::Array{Float64,1}, w::Workspace)
    q = w.q
    N = w.N
	LL = w.LL

	@assert LL == Int(length(x)/2)

    mysum1E = 0.0
	mysum1G = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += x[i] * x[i]
		mysum1G += x[i+LL] * x[i+LL]
    end
    mysum1E *= w.reg.lambdaJE
	mysum1G *= w.reg.lambdaJG

    mysum2E = 0.0
	mysum2G = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += x[i] * x[i]
		mysum2G += x[i+LL] * x[i+LL]
    end
    mysum2E *= 2*w.reg.lambdaHE
	mysum2G *= 2*w.reg.lambdaHG
    return mysum1E + mysum2E + mysum1G + mysum2G
end


##############################################################
# BETA OPTIMIZATION

function Beta2Grad(b::Float64, x::Matrix{Float64}, Z::AbstractArray{Int,2}, W::AbstractVector{Float64}, q::Int, N::Int)

	M = length(W)

	bvecener = zeros(Float64,q)
	vecener = zeros(Float64,q)
	g = 0.0
	g2 = 0.0

	for m = 1:M
		seq = view(Z,:,m)
		for site = 1:N
			fillvecene!(bvecener, vecener, x, site, seq, b, q, N)
			lnorm = logsumexp(bvecener)
			avener = sum( @. vecener * exp(bvecener - lnorm))
			av2ener = sum( @. (vecener * vecener) * exp.(bvecener - lnorm))
			g -= W[m]*( vecener[seq[site]] - avener )
			g2 += W[m]*(av2ener - avener*avener)
		end
	end

	return g,g2

end


function fillvecene!(bvecener::Vector{Float64},vecener::Vector{Float64},x::Matrix{Float64},site::Int,seq::AbstractArray{Int,1},b::Float64,q::Int,N::Int)

	q2 = q*q
	LL = (N-1)*q*q+q

    @inbounds for l = 1:q
		offset::Int=0
        bener::Float64 = 0.0
        ener::Float64 = 0.0

        for i = 1:site-1
            bener += b * x[offset + l + q * (seq[i]-1), site] + x[LL + offset + l + q * (seq[i]-1), site]
            ener += x[offset + l + q * (seq[i]-1), site]
			offset += q2
        end

        for i = site+1:N
			bener += b * x[offset + l + q * (seq[i]-1), site] + x[LL + offset + l + q * (seq[i]-1), site]
			ener += x[offset + l + q * (seq[i]-1), site]
			offset += q2
        end

        bener += b * x[offset + l,site] + x[LL + offset + l,site]
        ener += x[offset + l,site]

        bvecener[l] = bener
        vecener[l] = ener
    end

end
