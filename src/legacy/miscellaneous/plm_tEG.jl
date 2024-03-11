
# Computing G+t*E parameters
#INPUT: data with T rounds and vectors of beta with T elements (round 0 1 2 .. T-1)
#OUTPUT: G and E parameters (J and h)

function  plmdcat(data, beta::Vector{Float64};
						q::Int=20,
						N::Int=data.L,
						lambdaJE::Real=0.01,
						lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
						lambdaHG::Real=0.01,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-5,
						maxit::Int=10000,
						verbose::Bool=true,
						method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (data.T) != length(beta)
		throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, true, epsconv ,maxit)
	vecvar = PlmDCAt.set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

	q = vecvar[1].q
	N = vecvar[1].N
	q2 = vecvar[1].q2

	Jmat, pslike, fail = PlmDCAt.ParMinimize(alg,vecvar, beta, Jstart);
	pslvec = ComputePSL(Jmat,vecvar,beta);
	J_E,h_E,J_G,h_G = PlmDCAt.GEMat2TensNoGauge(Jmat,N,q);

	return (Jmat=Jmat, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike, pslv=pslvec)
end

function  plmdcat(vecvar::Vector{PlmVar}, beta::Vector{Float64};
						q::Int=20,
						N::Int=vecvar[1].N,
						epsconv::Real=1.0e-5,
						maxit::Int=10000,
						verbose::Bool=true,
						method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	T = size(vecvar,1)
	if T != length(beta)
		throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, true, epsconv ,maxit)

	q = vecvar[1].q
	N = vecvar[1].N
	q2 = vecvar[1].q2

	Jmat, pslike, fail = PlmDCAt.ParMinimize(alg,vecvar,beta,Jstart);
	pslvec = ComputePSL(Jmat,vecvar,beta);
	J_E,h_E,J_G,h_G = PlmDCAt.GEMat2TensNoGauge(Jmat,N,q);

	return (Jmat=Jmat, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike, pslv=pslvec)
end


######

#Computing pseudolikelihood  values for all combination of a grid of beta values specified by bgrid
#INPUT: data with T rounds
# bgrid has T-2 elements each of them a vector of values to be assessed
#OUTPUT: pseudolikelihood values for each grid point

function  plmdcagt(data, bgrid::Vector{Vector{Float64}};
						q::Int=20,
						N::Int=data.L,
						lambdaJE::Real=0.01,
                        lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
						lambdaHG::Real=0.01,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-5,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

    if (data.T)-2 != length(bgrid)
        throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
    end

    plmalg = PlmAlg(method, verbose, epsconv ,maxit)

	vecvar = set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

    psl = beta_grid(plmalg, vecvar, bgrid, Jstart)

    return psl=psl

end

#Grid function taking as input a PlmVar vector so to select specific rounds
function  plmdcagt(vecvar::Vector{PlmVar}, bgrid::Vector{Vector{Float64}};
						q::Int=20,
						N::Int=vecvar[1].N,
						epsconv::Real=1.0e-5,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	T = size(vecvar,1)

	if T-2 != length(bgrid)
		throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

    plmalg = PlmAlg(method, verbose, epsconv ,maxit)

    psl = beta_grid(plmalg, vecvar, bgrid, Jstart)

    return psl

end

function  plmdcagt_nr0(vecvar::Vector{PlmVar}, bgrid::Vector{Vector{Float64}};
						q::Int=20,
						N::Int=vecvar[1].N,
						epsconv::Real=1.0e-5,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	T = size(vecvar,1)

	if T-1 != length(bgrid)
		throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

    plmalg = PlmAlg(method, verbose, epsconv ,maxit)

    psl = bg_nr0(plmalg, vecvar, bgrid, Jstart)

    return psl

end

function beta_grid(alg::PlmAlg, var::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, Jstart::Matrix{Float64})

    T = size(var,1)
	q = var[1].q
	N = var[1].N
	q2 = var[1].q2
	Jtemp = zeros(Float64,2*((N-1)*q2+q),N)
	Jmat = zeros(Float64,2*((N-1)*q2+q),N)

	@assert (T-2) == length(bgrid)


    iter = collect.(Iterators.product(bgrid...))

	Jtemp = copy(Jstart)
    psl=Matrix{Float64}(undef,length(iter),T-1)
    b=Vector{Float64}(undef,T)

    counter = 0

    #@distributed
    for i in iter[:]
        counter += 1
        #pushfirst!(i,1.)
        b = vcat(0.,1.,i)
        Jmat, pslike, fail = ParMinimize(alg, var, b, Jtemp)
        if 1 in fail
            psl[counter,:] = vcat(i,NaN)
        else
            psl[counter,:] = vcat(i,sum(pslike))
        end
		Jtemp = copy(Jmat)
    end

    return psl

end

function bg_nr0(alg::PlmAlg, var::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, Jstart::Matrix{Float64})

    T = size(var,1)
	q = var[1].q
	N = var[1].N
	q2 = var[1].q2
	Jtemp = zeros(Float64,2*((N-1)*q2+q),N)
	Jmat = zeros(Float64,2*((N-1)*q2+q),N)

	@assert (T-1) == length(bgrid)


    iter = collect.(Iterators.product(bgrid...))

	Jtemp = copy(Jstart)
    psl=Matrix{Float64}(undef,length(iter),T)
    b=Vector{Float64}(undef,T)

    counter = 0

    #@distributed
    for i in iter[:]
        counter += 1
        #pushfirst!(i,1.)
        b = vcat(1.,i)
        Jmat, pslike, fail = ParMinimize(alg, var, b, Jtemp)
        if 1 in fail
            psl[counter,:] = vcat(i,NaN)
        else
            psl[counter,:] = vcat(i,sum(pslike))
        end
		Jtemp = copy(Jmat)
    end

    return psl

end

#Performing scan over regularization λ
function lambda_scan(data, bgrid::Vector{Vector{Float64}}, λvec::Vector{Float64};
					weight::Symbol=:counts,
					epsconv::Real=1.0e-5,
					maxit::Int=10000,
					verbose::Bool=true,
					method::Symbol=:LD_LBFGS)

 	if length(λvec) != length(bgrid)
		throw(DimensionMismatch("Number of λ's does not match temperatures"))
	end

	alg = PlmAlg(method, verbose, epsconv ,maxit)

	scan = Array{Array{Float64,2}}(undef,length(λvec))

	for j=1:length(λvec)

		λ = λvec[j]
		var = set_vecvar(data,lambdaJE=λ,lambdaHE=λ,lambdaJG=λ,lambdaHG=λ,weight=weight)
		beta = bgrid[j]
		scan[j] = Array{Float64,2}(undef,length(beta),3)

		q = var[1].q
		N = var[1].N
		q2 = var[1].q2
		Jtemp = zeros(Float64,2*((N-1)*q2+q),N)
		Jmat = zeros(Float64,2*((N-1)*q2+q),N)

		for i=1:length(beta)
	        b = vcat(0.,1.,beta[i])
	        Jmat, pslike, fail = ParMinimize(alg, var, b, Jtemp)
			l2 = 0.0
			for k=1:N
				l2 += L2norm_asymEG(Jmat[:,k],var[1])
			end

	        if 1 in fail
	            scan[j][i,:] = vcat(λ,beta[i],NaN)
	        else
	            scan[j][i,:] = vcat(λ,beta[i],sum(pslike)-l2)
	        end
			Jtemp = copy(Jmat)
		end

	end

	return scan
end

#########

# Optimization of beta and parameters of model G + t*E through Newton GD
# asymmetric pseudolikelihood method
#INPUT: data with T rounds
#OUTPUT: optimal beta, G and E parameters (J and h)

function newton_asym(data;
						q::Int=20,
						N::Int=data.L,
						lambdaJE::Real=0.01,
                        lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
                        lambdaHG::Real=0.01,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(2:data.T-1)),
						epsconv::Real=1.0e-6,
                        maxit::Int=10000,
						method::Symbol=:LD_LBFGS,
						verbose::Bool=true,
						betatol::Real=1.0e-3,
						maxit_glob::Int = 100,
						b_iter::Int = 1,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (data.T)-2 != length(bstart)
		throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

    counter::Int=0

	alg0 = PlmAlg(method,verbose,epsconv,10000)

	alg = PlmAlg(method,verbose,epsconv,maxit)

	var = set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

	q=var[1].q
	q2 = q*q
	N=var[1].N
	T = size(var,1)

    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
	bnew = zeros(Float64,T-2)

	#init
	Jmat = zeros(Float64,2*((N-1)*q2+q),N)
	#Jtemp = zeros(Float64,2*((N-1)*q2+q),N)
	Jtemp = copy(Jstart)
	J_E = zeros(Float64, q, q, N, N)
	h_E = zeros(Float64, q, N)
	J_G = zeros(Float64, q, q, N, N)
	h_G = zeros(Float64, q, N)
	fail_p = zeros(Int, N)

	pslike = 0.0

	Jmat, pslike, fail_p = ParMinimize(alg0, var, vcat(0.0,1.0,bold), Jtemp)

	Jtemp = copy(Jmat)

    while(any(Dbeta .> betatol) && counter < maxit_glob)

        bnew = BetaMinNewt(bold,var[3:end],Jtemp,b_iter)

		println(bnew)

		wisereldist!(Dbeta,bnew,bold)

		println(Dbeta)

		bold = copy(bnew)

		Jmat, pslike, fail_p = ParMinimize(alg, var, vcat(0.0,1.0,bold), Jtemp)

		if 1 in fail_p
			error("Pseudo-likelihood minimization failed")
		end

		Jtemp = copy(Jmat)

        counter += 1

    end

	J_E,h_E,J_G,h_G = GEMat2TensNoGauge(Jtemp,N,q)

    return (beta=bnew, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike, Jmat=Jtemp)


end

function newton_asym(var::Vector{PlmVar};
						bstart::Vector{Float64}=Float64.(collect(2:length(var)-1)),
						epsconv::Real=1.0e-5,
                        maxit::Int=10000,
						method::Symbol=:LD_LBFGS,
						verbose::Bool=true,
						betatol::Real=1.0e-3,
						maxit_glob::Int = 100,
						Jinit::Matrix{Float64}=zeros(Float64,2*((var[1].N-1)*var[1].q*var[1].q+var[1].q),var[1].N),
						b_iter::Int = 1)

	alg = PlmAlg(method,verbose,epsconv,maxit)

	q=var[1].q
	q2 = q*q
	N=var[1].N
	T = size(var,1)

	if T-2 != length(bstart)
		throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
	bnew = zeros(Float64,T-2)

	#init
	Jmat = zeros(Float64,2*((N-1)*q2+q),N)
	Jtemp = zeros(Float64,2*((N-1)*q2+q),N)
	J_E = zeros(Float64, q, q, N, N)
	h_E = zeros(Float64, q, N)
	J_G = zeros(Float64, q, q, N, N)
	h_G = zeros(Float64, q, N)
	fail_p = zeros(Int, N)

	counter::Int=0
	pslike = 0.0

	Jtemp = copy(Jinit)

    while(any(Dbeta .> betatol) && counter < maxit_glob)

        bnew = BetaMinNewt(bold,var[3:end],Jtemp,b_iter)

		println(bnew)

		wisereldist!(Dbeta,bnew,bold)

		println(Dbeta)

		bold = copy(bnew)

		Jmat, pslike, fail_p = ParMinimize(alg, var, vcat(0.0,1.0,bold), Jtemp)

		if 1 in fail_p
			error("Pseudo-likelihood minimization failed")
		end

		Jtemp = copy(Jmat)

        counter += 1

    end

	J_E,h_E,J_G,h_G = GEMat2TensNoGauge(Jtemp,N,q)

    return (beta=bnew, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike)


end

function plm_opt(data;
						q::Int=20,
						N::Int=data.L,
						lambdaJE::Real=0.01,
						lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
						lambdaHG::Real=0.01,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(2:data.T-1)),
						epsconv::Real=1.0e-10,
						maxit::Int=10000,
						method::Symbol=:LD_LBFGS,
						method_beta::Symbol=:LD_MMA,
						verbose::Bool=true,
						betatol::Real=5.0e-4,
						maxit_glob::Int = 100,
						b_iter::Int = 4,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N),
                        epsconv_beta::Real=1.0e-15)

	if (data.T)-2 != length(bstart)
		throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

	counter::Int=0

	alg = PlmAlg(method,verbose,epsconv,maxit)

	algbeta = PlmAlg(method_beta,verbose,epsconv_beta,b_iter)

	var = set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

	q2 = q*q
	T = size(var,1)

	Dbeta = ones(Float64,T-2)
	bold = copy(bstart)
	bnew = zeros(Float64,T-2)

	#init
	Jmat = zeros(Float64,2*((N-1)*q2+q),N)
	#Jtemp = zeros(Float64,2*((N-1)*q2+q),N)
	Jtemp = copy(Jstart)
	J_E = zeros(Float64, q, q, N, N)
	h_E = zeros(Float64, q, N)
	J_G = zeros(Float64, q, q, N, N)
	h_G = zeros(Float64, q, N)
	fail_p = zeros(Int, N)
	fail_b = zeros(Int,T-2)

	pslike = 0.0

	Jmat, pslike, fail_p = ParMinimize(alg, var, vcat(0.0,1.0,bold), Jtemp)

	Jtemp = copy(Jmat)

    while(any(Dbeta .> betatol) && counter < maxit_glob)

        bnew, fail_b = BetaMinimize(bold,algbeta,var[3:end],Jtemp)

        if 1 in fail_b
            error("Pseudo-likelihood minimization failed")
        end

		println(bnew)

		wisereldist!(Dbeta,bnew,bold)

		println(Dbeta)

		bold = copy(bnew)

		Jmat, pslike, fail_p = ParMinimize(alg, var, vcat(0.0,1.0,bold), Jtemp)

		if 1 in fail_p
			error("Pseudo-likelihood minimization failed")
		end

		Jtemp = copy(Jmat)

        counter += 1

    end

	J_E,h_E,J_G,h_G = GEMat2TensNoGauge(Jtemp,N,q)

    return (beta=bnew, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike, Jmat=Jtemp)


end

########################################################

function ParMinimize(alg::PlmAlg, var::Vector{PlmVar}, beta::Vector{Float64},
	Jtemp::Matrix{Float64})

	@assert length(var) == length(beta)

    LL = (var[1].N - 1) * var[1].q2 + var[1].q
    x0 = zeros(Float64,2*LL,var[1].N)
    vecps = SharedArray{Float64}(var[1].N)
    fail = SharedArray{Int}(var[1].N)

	@assert size(Jtemp,1) == 2*LL

	x0 = copy(Jtemp)

    Jmat = @distributed hcat for site=1:var[1].N #1:12
        opt = Opt(alg.method, length(x0[:,site]))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        min_objective!(opt, (x,g)->optimfunwrapper(x, g, site, var, beta))
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
	plmvar::Vector{PlmVar},	beta::Vector{Float64})

    LL = Int(length(vecJ)/2)
    q2 = plmvar[1].q2
    q = plmvar[1].q
    N = plmvar[1].N
	T = size(plmvar,1)

	@assert LL == (N-1)*q2+q

    pseudolike = 0.0

    for i=1:LL-q
	    grad[i] = 2.0 * plmvar[1].lambdaJE * vecJ[i]
	end
	for i=(LL-q+1):LL
	    grad[i] = 4.0 * plmvar[1].lambdaHE * vecJ[i]
	end

	for i=LL+1:2*LL-q
		grad[i] = 2.0 * plmvar[1].lambdaJG * vecJ[i]
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
	pseudolike += L2norm_asymEG(vecJ, plmvar[1])
    return pseudolike
end



function fillvecene!(vecene::Array{Float64,1},vecJ::Array{Float64,1},site::Int,a::Int,var::PlmVar,b::Float64)
	LL = Int(length(vecJ)/2)
	q=var.q
    N=var.N
    q2 = q*q
    Z = sdata(var.Z)

    @inbounds for l = 1:q
        offset::Int = 0
        scra::Float64 = 0.0
        for i = 1:site-1 # Begin sum_i \neq site j
            scra += b*vecJ[offset + l + q * (Z[i,a]-1)] + vecJ[LL + offset + l + q * (Z[i,a]-1)]
            offset += q2
        end
        # skipping sum over residue site
    	for i = site+1:N
			scra += b*vecJ[offset + l + q * (Z[i,a]-1)] + vecJ[LL + offset + l + q * (Z[i,a]-1)]
            offset += q2
        end # End sum_i \neq site j
        scra +=  b*vecJ[offset + l] + vecJ[LL+offset+l] # sum H
        vecene[l] = scra
    end
end

function L2norm_asymEG(vec::Array{Float64,1}, plmvar::PlmVar)
    q = plmvar.q
    N = plmvar.N
    lambdaJE = plmvar.lambdaJE
    lambdaHE = plmvar.lambdaHE
	lambdaJG = plmvar.lambdaJG
	lambdaHG = plmvar.lambdaHG

    LL = Int(length(vec)/2)

    mysum1E = 0.0
	mysum1G = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1E += vec[i] * vec[i]
		mysum1G += vec[LL+i]*vec[LL+i]
    end
    mysum1E *= lambdaJE
	mysum1G *= lambdaJG

    mysum2E = 0.0
	mysum2G = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2E += vec[i] * vec[i]
		mysum2G += vec[LL+i]*vec[LL+i]
    end
    mysum2E *= 2*lambdaHE
	mysum2G *= 2*lambdaHG
    return mysum1E+mysum1G+mysum2E+mysum2G
end

#Fuction which minimizes psl performing steps over β according to Newton GD
function BetaMinNewt(b_old::Vector{Float64},var::Vector{PlmVar},Jmat::Matrix{Float64},b_iter::Int)

	b_init = copy(b_old)
	bnew = zeros(Float64,length(b_init))
	grad = zeros(Float64,length(b_init))
	grad2 = zeros(Float64,length(b_init))

	T = size(var,1)

	for iter=1:b_iter
		for t=1:T
			grad[t],grad2[t]=Beta2Grad(b_init[t],var[t],Jmat)
			bnew[t] = b_init[t] - grad[t]/grad2[t]
			b_init[t] = bnew[t]
		end
		println(grad)
		println(grad2)
	end


	return bnew

end

#Compute first and second derivative with respect to β
function Beta2Grad(b::Float64,var::PlmVar,Jmat::Matrix{Float64})

	N = var.N
	q = var.q
	M = var.M
	Z = sdata(var.Z)
	W = sdata(var.W)

	bvecener = zeros(Float64,q)
	vecener = zeros(Float64,q)
	g = 0.0
	g2 = 0.0

	for site=1:N
		for a=1:M
			fillvecene!(bvecener,vecener,site,a,N,q,Z,b,Jmat)
			norm = sumexp(bvecener)
			avener = sum( vecener .* exp.(bvecener))/norm
			av2ener = sum( (vecener .* vecener) .* exp.(bvecener))/norm
			g += -W[a]*( vecener[Z[site,a]] - avener )
			g2 += W[a]*(av2ener - avener*avener)
		end
	end

	return g,g2

end

function fillvecene!(bvecener::Vector{Float64},vecener::Vector{Float64},site::Int,a::Int,N::Int,q::Int,Z::Matrix{Int},b::Float64,Jmat::Matrix{Float64})

	q2 = q*q
	LL::Int=(N-1)*q2+q

    @inbounds for l = 1:q
		offset::Int=0
        bener::Float64 = 0.0
        ener::Float64 = 0.0

        for i=1:site-1
            bener += b*Jmat[l+offset+q*(Z[i,a]-1),site]+Jmat[LL+l+offset+q*(Z[i,a]-1),site]
            ener += Jmat[l+offset+q*(Z[i,a]-1),site]
			offset += q2
        end

        for i=site+1:N
			bener += b*Jmat[l+offset+q*(Z[i,a]-1),site]+Jmat[LL+l+offset+q*(Z[i,a]-1),site]
			ener += Jmat[l+offset+q*(Z[i,a]-1),site]
			offset += q2
        end

        bener += b*Jmat[offset+l,site]+Jmat[LL+offset+l,site]
        ener += Jmat[offset+l,site]

        bvecener[l] = bener
        vecener[l] = ener
    end

end

function BetaMinimize(beta::Vector{Float64},alg::PlmAlg,var::Vector{PlmVar},Jmat::Matrix{Float64})

    T=size(var,1)
    pslike=zeros(Float64,T)
    x0vec = copy(beta)
    bnew=zeros(Float64,T)
    fail = zeros(Int,T)

    for t=1:T
        x0 = [x0vec[t]]
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        lower_bounds!(opt, [0.0])
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,var[t],Jmat))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("round = %d\t pl = %.4f\t time = %.4f\t", t+2, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        if ret==:FAILURE
            fail[t] = 1
        end
        pslike[t]=minf
        bnew[t] = minx[1]
    end
    return bnew, fail #sum(pslike), fail
end

function PLbetaGrad!(beta::Vector{Float64},grad::Vector{Float64},var::PlmVar,Jmat::Matrix{Float64})

	grad[1] = 0.0

    N = var.N
    q = var.q
    pseudolike = 0.0

    M = var.M
    Z = sdata(var.Z)
    W = sdata(var.W)

    bvecener = zeros(Float64,q)
    vecener = zeros(Float64,q)

	for site=1:N
		for a=1:M
			fillvecene!(bvecener,vecener,site,a,N,q,Z,beta[1],Jmat)
			norm = sumexp(bvecener)
			avener = sum( vecener .* exp.(bvecener))/norm
			grad[1] -= W[a]*(vecener[Z[site,a]]-avener)
			pseudolike -= W[a] * (bvecener[Z[site,a]] - log(norm))
			#lnorm = log(sumexp(bvecener))
			#pseudolike -= W[a] * (bvecener[Z[site,a]] - lnorm)
			#grad[1] -= W[a]*( vecener[Z[site,a]] - sum( vecener .* exp.(bvecener .- lnorm)) )
		end
    end

    @printf("grad = %.4f\tpsl=%.4f\n",grad[1],pseudolike)

    return pseudolike

end

#Function minimizing psl with respect to β automatically


#Funzioni per il calcolo automatico del gradiente con Zygote
#function f(β,data::PlmVar,par::Matrix{Float64})
#	function ff(β)
#		psl(β,data,par)
#	end
#	return ff'(β)
#end

#function psl(β,data::PlmVar,par::Matrix{Float64})

#	pseudolike = 0.0

#	N = data.N
#	q = data.q
#	M = data.M
#	Z = sdata(data.Z)
#	W = sdata(data.W)
#	E = zeros(Float64,N,M,q)
#	G = zeros(Float64,N,M,q)


#	E,G = EnerMatrix(data,par)


#	for site=1:N
#		for a=1:M
#			lnorm = log(sum(exp.(β*E[site,a,:].+G[site,a,:])))
#			pseudolike -= W[a] * (β*E[site,a,Z[site,a]]+G[site,a,Z[site,a]] - lnorm)
#		end
#	end

#	return pseudolike

#end

#function fillvecene!(vecener::Vector{Float64},site::Int,N::Int,q::Int,a::Int,Z::Matrix{Int},β,
#	J_E::Array{Float64,4},h_E::Array{Float64,2},J_G::Array{Float64,4},h_G::Array{Float64,2})

#    @inbounds for l = 1:q
#        ener::Float64 = 0.0

#        for i=1:site-1
#            ener += β*J_E[l,Z[i,a],site,i]+J_G[l,Z[i,a],site,i]
#        end

#        for i=site+1:N
#            ener += β*J_E[l,Z[i,a],site,i]+J_G[l,Z[i,a],site,i]
#        end

#        ener += β*h_E[l,site]+h_G[l,site]

#        vecener[l] = ener
#    end

#end

#function fillvecene(J::Array{Float64,4},h::Array{Float64,2},N::Int,q::Int,site::Int,a::Int,Z::Array{Int64,2})

#    vecener = zeros(Float64,q)

#    for l=1:q
#        for i=1:site-1
#            vecener[l] += J[l,Z[i,a],site,i]
#        end

#        for i=site+1:N
#            vecener[l] += J[l,Z[i,a],site,i]
#        end

#        vecener[l] += h[l,site]

#    end

#    return vecener

#end

#function EnerMatrix(var::PlmVar, Jmat::Matrix{Float64})

#    N = var.N
#    q = var.q
#    q2=q*q
#	M=var.M

#	E = zeros(Float64,N,M,q)
#	G = zeros(Float64,N,M,q)

#	Je,he,Jg,hg = GEMat2Tensor(Jmat,N,q)

#	Z = sdata(var.Z)
#    for site=1:N
#        for a=1:var.M
#            E[site,a,:] = fillvecene(Je,he,N,q,site,a,Z)
#            G[site,a,:] = fillvecene(Jg,hg,N,q,site,a,Z)
#		end
#    end

#	return E,G

#end
