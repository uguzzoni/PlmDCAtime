#Computing t*E+ν*M+G parameters
#INPUT: data with T rounds, vectors of beta and ν with T elements (rounds 0 1 2 .. T-1)
#OUTPUT: G, M and E parameters (J and h)

function  plmdcatv(data, beta::Vector{Float64}, nu::Vector{Float64};
						q::Int=data.A,
						N::Int=data.L,
						lambdaJE::Real=0.01,
						lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
						lambdaHG::Real=0.01,
						lambdaJM::Real=0.01,
						lambdaHM::Real=0.01,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-5,
						maxit::Int=10000,
						verbose::Bool=true,
						method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,3*((N-1) *q*q + q),N))

	if (data.T) != length(beta) || data.T != length(nu)
		throw(DimensionMismatch("Number of temperature or frequency vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, verbose, epsconv ,maxit)
	vecvar = PlmDCAt.set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,lambdaJM=lambdaJM,lambdaHM=lambdaHM,weight=weight)

	q = vecvar[1].q
	N = vecvar[1].N
	q2 = vecvar[1].q2

	Jmat, pslike, fail = PlmDCAt.ParMinimize(alg,vecvar,beta,nu,Jstart);
	J_E,h_E,J_G,h_G,J_M,h_M = PlmDCAt.GMEMat2TensNoGauge(Jmat,N,q);

	return (Jmat=Jmat, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, J_M=J_M, h_M=h_M, pslv=pslike)

end

function  plmdcatv(vecvar::Vector{PlmVar}, beta::Vector{Float64}, nu::Vector{Float64};
						q::Int=20,
						N::Int=vecvar[1].N,
						epsconv::Real=1.0e-5,
						maxit::Int=10000,
						verbose::Bool=true,
						method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,3*((N-1)*q*q+q),N))

	T = size(vecvar,1)
	if T != length(beta) || T != length(nu)
		throw(DimensionMismatch("Number of temperature or frequency vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, verbose, epsconv ,maxit)

	q = vecvar[1].q
	N = vecvar[1].N
	q2 = vecvar[1].q2

	Jmat, pslike, fail = PlmDCAt.ParMinimize(alg,vecvar,beta,nu,Jstart);
	J_E,h_E,J_G,h_G,J_M,h_M = PlmDCAt.GMEMat2TensNoGauge(Jmat,N,q);

	return (Jmat=Jmat, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, J_M=J_M, h_M=h_M, pslv=pslike)
end


function ParMinimize(alg::PlmAlg, var::Vector{PlmVar}, beta::Vector{Float64}, nu::Vector{Float64},
	Jtemp::Matrix{Float64})

	@assert length(var) == length(beta) == length(nu)

    LL = (var[1].N - 1) * var[1].q2 + var[1].q
    x0 = zeros(Float64,3*LL,var[1].N)
    vecps = SharedArray{Float64}(var[1].N)
    fail = SharedArray{Int}(var[1].N)

	@assert size(Jtemp,1) == 3*LL

	x0 = copy(Jtemp)

    Jmat = @distributed hcat for site=1:var[1].N #1:12
        opt = Opt(alg.method, length(x0[:,site]))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        min_objective!(opt, (x,g)->optimfunwrapper(x, g, site, var, beta, nu))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0[:,site])
        alg.verbose && @printf("site = %d\t pl = %.4f\t time = %.4f\t", site, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        if ret==:FAILURE
            fail[site] = 1
			error("Pseudo-likelihood minimization failed")
        end
        vecps[site] = minf
        minx
    end

    return Jmat, vecps, fail
end



function PLsiteAndGrad!(vecJ::Vector{Float64}, grad::Vector{Float64},site::Int,
	plmvar::Vector{PlmVar},	beta::Vector{Float64}, nu::Vector{Float64})

    LL = Int(length(vecJ)/3)
    q2 = plmvar[1].q2
    q = plmvar[1].q
    N = plmvar[1].N
	T = size(plmvar,1)

	@assert LL == (N-1)*q2+q

    for i=1:LL-q
	    grad[i] = 2.0 * plmvar[1].lambdaJE * vecJ[i]
	end
	for i=(LL-q+1):LL
	    grad[i] = 4.0 * plmvar[1].lambdaHE * vecJ[i]
	end

	for i=LL+1:2*LL-q
		grad[i] = 2.0 * plmvar[1].lambdaJG * vecJ[i]
	end
	for i=(2*LL-q+1):2*LL
		grad[i] = 4.0 * plmvar[1].lambdaHG * vecJ[i]
	end

	for i=2*LL+1:3*LL-q
		grad[i] = 2.0 * plmvar[1].lambdaJM * vecJ[i]
	end
	for i=(3*LL-q+1):3*LL
		grad[i] = 4.0 * plmvar[1].lambdaHM * vecJ[i]
	end

	vecene = zeros(Float64,q)
	expvecenesumnorm = zeros(Float64,q)
	pseudolike = 0.0
	lnorm = 0.0
	xx = Array{Float64}(undef, LL);

    for t = 1:T

        M = plmvar[t].M
	    IdxZ = sdata(plmvar[t].IdxZ)
		Zsite = view(plmvar[t].Z,site,:);
    	W = sdata(plmvar[t].W)

		#xx .=  beta[t] .* view(vecJ,1:LL) .+ view(vecJ,(LL+1):(2*LL)) .+ nu[t] .* view(vecJ,(2*LL+1):(3*LL))
		xx .= @. beta[t] * vecJ[1:LL] + vecJ[LL+1:2*LL] + nu[t] * vecJ[2*LL+1:end]
		#xx .=  beta[t] * view(vecJ,1:LL) + view(vecJ,(LL+1):(2*LL)) + nu[t] * view(vecJ,(2*LL+1):(3*LL))

	   @inbounds for a = 1:M

		    #fillvecene!(vecene, xx, site, view(IdxZ,:,a), q, N)
			fillvecene!(vecene, xx, site, a, plmvar[t])

			lnorm = logsumexp(vecene)
			#lnorm = log(sumexp(vecene))
	        expvecenesumnorm .= @. exp(vecene - lnorm)
			#expvecenesumnorm .= exp.(vecene .- lnorm)

			pseudolike -= W[a] * (vecene[Zsite[a]] - lnorm)

	        @simd for i = 1:site-1
	            @simd for s = 1:q
					grad[ IdxZ[i,a] + s  ] += W[a] * beta[t] * expvecenesumnorm[s]
					grad[ LL + IdxZ[i,a] + s ] += W[a] * expvecenesumnorm[s]
					grad[ 2*LL + IdxZ[i,a] + s  ] += W[a] * nu[t] * expvecenesumnorm[s]

				end

				grad[  IdxZ[i,a] + Zsite[a] ] -= W[a] * beta[t]
				grad[ LL +  IdxZ[i,a] + Zsite[a] ] -= W[a]
				grad[ 2*LL +  IdxZ[i,a] + Zsite[a] ] -= W[a] * nu[t]

	        end
            @simd for i = site+1:N
	            @simd for s = 1:q
					grad[ IdxZ[i,a] - q2 + s ] += W[a] * beta[t] * expvecenesumnorm[s]
					grad[ LL + IdxZ[i,a] - q2 + s ] += W[a] * expvecenesumnorm[s]
					grad[ 2*LL + IdxZ[i,a] - q2 + s ] += W[a] * nu[t] * expvecenesumnorm[s]

	            end

				grad[ IdxZ[i,a] - q2 + Zsite[a] ] -= W[a] * beta[t]
				grad[ LL + IdxZ[i,a] - q2 + Zsite[a] ] -= W[a]
				grad[ 2*LL + IdxZ[i,a] - q2 + Zsite[a] ] -= W[a] * nu[t]

	        end
	        @simd for s = 1:q
	            grad[ (N-1) * q2 + s ] += W[a] * beta[t] * expvecenesumnorm[s]
				grad[ LL + (N-1) * q2 + s ] += W[a] * expvecenesumnorm[s]
				grad[ 2*LL + (N-1) * q2 + s ] += W[a] * nu[t] * expvecenesumnorm[s]
	        end

			grad[ (N-1) * q2 + Zsite[a] ] -= W[a] * beta[t]
			grad[ LL + (N-1) * q2 + Zsite[a] ] -= W[a]
			grad[ 2*LL + (N-1) * q2 + Zsite[a] ] -= W[a] * nu[t]

	    end

	end
	pseudolike += L2norm_asymEGM(vecJ, plmvar[1])
    return pseudolike
end

# function fillvecene!(vecene::Vector{Float64}, x::Vector{Float64}, site::Int, IdxSeq::AbstractArray{Int,1}, q::Int, N::Int)
#
# 	q2=q*q
#     @inbounds for l = 1:q
#         scra::Float64 = 0.0
#         @simd for i = 1:site-1 # Begin sum_i \neq site J
#             scra += x[IdxSeq[i] + l]
#         end
#         # skipping sum over residue site
#     	@simd for i = site+1:N
#             scra +=  x[IdxSeq[i] - q2 + l]
#         end # End sum_i \neq site J
#         scra +=  x[(N-1)*q2 + l] # sum H
#         vecene[l] = scra
#     end
#
# end


function fillvecene!(vecene::Vector{Float64}, x::Vector{Float64}, site::Int,  a::Int, var::PlmVar)

	IdxSeq = sdata(var.IdxZ)
	q=var.q
	N=var.N
	q2=var.q2

    @inbounds for l = 1:q
        scra::Float64 = 0.0
        @simd for i = 1:site-1 # Begin sum_i \neq site J
            scra += x[IdxSeq[i,a] + l]
        end
        # skipping sum over residue site
    	@simd for i = site+1:N
            scra +=  x[IdxSeq[i,a] - q2 + l]
        end # End sum_i \neq site J
        scra +=  x[(N-1)*q2 + l] # sum H
        vecene[l] = scra
    end

end


function L2norm_asymEGM(vec::Array{Float64,1}, plmvar::PlmVar)
    q = plmvar.q
    N = plmvar.N
    lambdaJE = plmvar.lambdaJE
    lambdaHE = plmvar.lambdaHE
	lambdaJG = plmvar.lambdaJG
	lambdaHG = plmvar.lambdaHG
	lambdaJM = plmvar.lambdaJM
	lambdaHM = plmvar.lambdaHM

    LL = Int(length(vec)/3)

    mysum1E = 0.0
	mysum1G = 0.0
	mysum1M = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += vec[i] * vec[i]
		mysum1G += vec[LL+i]*vec[LL+i]
		mysum1M += vec[2*LL+i]*vec[2*LL+i]
    end
    mysum1E *= lambdaJE
	mysum1G *= lambdaJG
	mysum1M *= lambdaJM

    mysum2E = 0.0
	mysum2G = 0.0
	mysum2M = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += vec[i] * vec[i]
		mysum2G += vec[LL+i]*vec[LL+i]
		mysum2M += vec[2*LL+i]*vec[2*LL+i]
    end
    mysum2E *= 2*lambdaHE
	mysum2G *= 2*lambdaHG
	mysum2M *= 2*lambdaHM
    return mysum1E+mysum1G+mysum1M+mysum2E+mysum2G+mysum2M
end
