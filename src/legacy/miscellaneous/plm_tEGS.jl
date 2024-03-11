#Functions which perform optimization including a cross regularization term λEG*||J_E-J_G||^2 

function  plmdcat_ls(data, beta::Vector{Float64}, l_sc::Float64;
						lambdaJE::Real=0.01,
						lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
						lambdaHG::Real=0.01,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-5,
						maxit::Int=10000,
						verbose::Bool=true,
						method::Symbol=:LD_LBFGS)

	if (data.T) != length(beta)
		throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, true, epsconv ,maxit)
	vecvar = PlmDCAt.set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

	q = vecvar[1].q
	N = vecvar[1].N
	q2 = vecvar[1].q2
	Jmat, pslike, fail = PlmDCAt.ParMinimize(alg,vecvar,beta,zeros(Float64,2*((N-1)*q2+q),N),l_sc);
	J_E,h_E,J_G,h_G = PlmDCAt.GEMat2Tensor(Jmat,N,q);

	return (J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike)
end

function ParMinimize(alg::PlmAlg,var::Vector{PlmVar}, beta::Vector{Float64},
	Jtemp::Matrix{Float64}, l::Float64)

	@assert length(var) == length(beta)

    LL = (var[1].N - 1) * var[1].q2 + var[1].q
    x0 = zeros(Float64,2*LL,var[1].N)
    vecps = SharedArray{Float64}(var[1].N)
    fail = SharedArray{Int}(var[1].N)
	#x0 = zeros(Float64,2*LL)

	@assert size(Jtemp,1) == 2*LL

	x0 = copy(Jtemp)

    Jmat = @distributed hcat for site=1:var[1].N #1:12
        opt = Opt(alg.method, length(x0[:,site]))
		#opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        min_objective!(opt, (x,g)->optimfunwrapper(x, g, site, var, beta,l))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0[:,site])
		#elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
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
	plmvar::Vector{PlmVar},	beta::Vector{Float64}, l::Float64)

    LL = Int(length(vecJ)/2)
    q2 = plmvar[1].q2
    q = plmvar[1].q
    N = plmvar[1].N
	T=size(plmvar,1)

	@assert LL == (N-1)*q2+q

    pseudolike = 0.0

    for i=1:LL-q
	    grad[i] = 2.0 * (plmvar[1].lambdaJE-l) * vecJ[i] + 2*l*vecJ[i+LL]
	end
	for i=(LL-q+1):LL
	    grad[i] = 4.0 * plmvar[1].lambdaHE * vecJ[i]
	end

	for i=LL+1:2*LL-q
		grad[i] = 2.0 * (plmvar[1].lambdaJG-l) * vecJ[i] + 2*l*vecJ[i-LL]
		#grad[i] = 0.
	end
	for i=(2*LL-q+1):2*LL
		grad[i] = 4.0 * plmvar[1].lambdaHG * vecJ[i]
	end

	vecene = zeros(Float64,q)
	expvecenesumnorm = zeros(Float64,q)

    for t = 1:T

        M = plmvar[t].M
    	Z = sdata(plmvar[t].Z)
    	W = sdata(plmvar[t].W)

	   @inbounds for a = 1:M

 			fillvecene!(vecene,vecJ,site,a,plmvar[t],beta[t])
	        lnorm = log(sumexp(vecene))
	        expvecenesumnorm .= exp.(vecene .- lnorm)
	        pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)
	        offset = 0
	        for i = 1:site-1
	            @simd for s = 1:q
	                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * beta[t] * expvecenesumnorm[s]
					grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * expvecenesumnorm[s]
					#grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] = 0.0
	            end
	            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * beta[t]
				grad[ LL + offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a]
				#grad[LL + offset + Z[site,a] + q * ( Z[i,a] - 1 )] = 0.0
	            offset += q2
	        end
            for i = site+1:N
	            @simd for s = 1:q
	                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * beta[t] * expvecenesumnorm[s]
					grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * expvecenesumnorm[s]
					#grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] = 0.0
	            end
	            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * beta[t]
				grad[ LL + offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a]
				#grad[LL + offset + Z[site,a] + q * ( Z[i,a] - 1 )] = 0.0
	            offset += q2
	        end
	        @simd for s = 1:q
	            grad[ offset + s ] += W[a] * beta[t] * expvecenesumnorm[s]
				grad[ LL + offset + s ] += W[a] * expvecenesumnorm[s]
	        end
			grad[ offset + Z[site,a] ] -= W[a] * beta[t]
			grad[ LL + offset + Z[site,a] ] -= W[a]
	    end

	end
	pseudolike += L2norm_asymEG(vecJ, plmvar[1],l)
    return pseudolike
end

function L2norm_asymEG(vec::Array{Float64,1}, plmvar::PlmVar, l::Float64)
    q = plmvar.q
    N = plmvar.N
    lambdaJE = plmvar.lambdaJE
    lambdaHE = plmvar.lambdaHE
	lambdaJG = plmvar.lambdaJG
	lambdaHG = plmvar.lambdaHG

    LL = Int(length(vec)/2)

    mysum1E = 0.0
	mysum1G = 0.0
	mysum1EG = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += vec[i] * vec[i]
		mysum1G += vec[LL+i]*vec[LL+i]
		mysum1EG += vec[i]*vec[i+LL]
    end
    mysum1E *= (lambdaJE-l)
	mysum1G *= (lambdaJG-l)
	mysum1EG *= 2*l

    mysum2E = 0.0
	mysum2G = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += vec[i] * vec[i]
		mysum2G += vec[LL+i]*vec[LL+i]
    end
    mysum2E *= 2*lambdaHE
	mysum2G *= 2*lambdaHG
    return mysum1E+mysum1G+mysum1EG+mysum2E+mysum2G
end
