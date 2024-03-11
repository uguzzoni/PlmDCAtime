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
	beta::Vector{Float64}, vv::Vector{Float64})


   LL = w.LL
   q = w.q
   q2 = q*q
   N = w.N
   T = w.T

   @assert T == length(beta)

   for i=1:LL-q
	   grad[i] = 2.0 * w.reg.lambdaJE * x[i]
	   #grad[i] = 0.0
   end

   for i=(LL-q+1):LL
	   grad[i] = 4.0 * w.reg.lambdaHE * x[i]
	   #grad[i] = 0.0
   end

   for i=LL+1:2*LL-q
	   grad[i] = 2.0 * w.reg.lambdaJG * x[i]
		#grad[i] = 0.0
   end
   for i=(2*LL-q+1):2*LL
	   grad[i] = 4.0 * w.reg.lambdaHG * x[i]
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
	   v = vv[t]

	   xx .= @. b * x[1:LL] + v * x[LL+1:2*LL]

	  @inbounds for a = 1:M

		   fillvecene!(vecene, xx, site, view(IdxZ,:,a), q, N)

		   lnorm = logsumexp(vecene)
		   expvecenesumnorm .= @. exp(vecene - lnorm)

		   pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)

		   @simd for i = 1:site-1
			   @simd for s = 1:q
				   grad[ IdxZ[i,a] + s  ] += W[a] * b * expvecenesumnorm[s]
				   grad[ LL + IdxZ[i,a] + s ] += W[a] * v * expvecenesumnorm[s]

			   end

			   grad[ IdxZ[i,a] + Z[site,a] ] -= W[a] * b
			   grad[ LL +  IdxZ[i,a] + Z[site,a] ] -= W[a] * v

		   end
		   @simd for i = site+1:N
			   @simd for s = 1:q
				   grad[ IdxZ[i,a] - q2 + s ] += W[a] * b * expvecenesumnorm[s]
				   grad[ LL + IdxZ[i,a] - q2 + s ] += W[a]* v * expvecenesumnorm[s]

			   end

			   grad[ IdxZ[i,a] - q2 + Z[site,a] ] -= W[a] * b
			   grad[ LL + IdxZ[i,a] - q2 + Z[site,a] ] -= W[a] * v

		   end
		   @simd for s = 1:q
			   grad[ (N-1) * q2 + s ] += W[a] * b * expvecenesumnorm[s]
			   grad[ LL + (N-1) * q2 + s ] += W[a] * v * expvecenesumnorm[s]

		   end

		   grad[ (N-1) * q2 + Z[site,a] ] -= W[a] * b
		   grad[ LL + (N-1) * q2 + Z[site,a] ] -= W[a] * v

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
