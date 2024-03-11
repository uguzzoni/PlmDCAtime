#Looks for the optimal temperatures exploring a grid of values
function  plmdcagt(data, bedge::Vector{Vector{Float64}};
                        min_separation::Int = 0,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS)

    if (data.T)-2 != length(bedge)
        throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
    end

    plmalg = PlmAlg(method, verbose, epsconv ,maxit)

	vecvar = set_vecvar(data,lambdaJ=lambdaJ,lambdaH=lambdaH,weight=weight)

	Jmat0, psl0 = MinimizePLAsym(plmalg, vecvar[1]) #round 0
	J0, h0 = FieldsMatrix2Tensor(Jmat0, vecvar[1].N, vecvar[1].q)

    psl = beta_grid(plmalg, vecvar[2:end], bedge, J0, h0)

    return psl

end


function beta_grid(alg::PlmAlg, var::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, J0::Array{Float64,4}, h0::Array{Float64,2})

    T = size(var,1)
	q = var[1].q
	N = var[1].N
	q2 = var[1].q2
	Jtemp = zeros(Float64,(N-1)*q2+q,N)
	Jmat = zeros(Float64,(N-1)*q2+q,N)

	@assert (T-1) == length(bgrid)


    iter = collect.(Iterators.product(bgrid...))

    psl=Matrix{Float64}(undef,length(iter),T)
    b=Vector{Float64}(undef,T)

    counter = 0

    #@distributed
    for i in iter[:]
        counter += 1
        #pushfirst!(i,1.)
        b = vcat(1.,i)
        Jmat, pslike, fail = ParMinimize(alg, var, b, Jtemp, J0, h0)
        if 1 in fail
            psl[counter,:] = vcat(i,NaN)
        else
            psl[counter,:] = vcat(i,sum(pslike))
        end
		Jtemp = copy(Jmat)
    end

    return psl

end

function plm_opt(data;
                        min_separation::Int = 0,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        lambdaB::Real=0.0,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(2:data.T-1)),
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
						method_par::Symbol=:LD_LBFGS,
                        epsconvbeta::Real=1.0e-5,
						betatol::Real=1.0e-3,
                        maxitbeta::Int=10,
                        verbose::Bool=true,
						method_beta::Symbol=:LD_MMA,
						maxit_glob::Int = 100)

	if (data.T)-2 != length(bstart)
	        throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

    counter::Int=0

    plmalgpar = PlmAlg(method_par,verbose,epsconv,maxit)

	plmalgbeta = PlmAlg(method_beta,verbose,epsconvbeta,betatol,maxitbeta)

    vecvar = set_vecvar(data,lambdaJ=lambdaJ,lambdaH=lambdaH,lambdaB=lambdaB,weight=weight)

	q=vecvar[1].q
	N=vecvar[1].N

    T = size(vecvar,1)
    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
	bnew = zeros(Float64,T-2)

	#init
    J = zeros(Float64, q, q, N, N)
    h = zeros(Float64, q, N)
    fail_p = zeros(Int, N)
    fail_b = zeros(Int, T-2)

    pslike = 0.0

    G = Array{Array{Float64,3}}(undef, T-2)
    E0 = Array{Array{Float64,3}}(undef, T-2)
    for t=1:T-2
        G[t] = zeros(Float64, N, vecvar[t+2].M, q)
        E0[t] = zeros(Float64, N, vecvar[t+2].M, q)
    end

    Jmat0, psl0 = MinimizePLAsym(plmalgpar, vecvar[1]) #round 0
	J0, h0 = FieldsMatrix2Tensor(Jmat0, N, q)

    println("Round 0 computation ended")

    while(any(Dbeta .> betatol) && counter < maxit_glob)

        J, h = Opt_Fields(plmalgpar, vecvar[2:end], vcat(1.,bold), J0, h0) #all rounds apart round 0

        EnerMatrix!(G, E0, vecvar[3:end], J , h, J0, h0) #all rounds apart rounds 0 amd 1

        bnew, pslike, fail_b = BetaMinimize(plmalgbeta, vecvar[3:end], bold, G, E0) #all rounds apart rounds 0 and 1

        if 1 in fail_b
            error("Pseudo-likelihood minimization failed")
        end

        wisereldist!(Dbeta,bnew,bold)

        println(Dbeta)

        bold = copy(bnew)

        counter += 1

    end

    J, h = Opt_Fields(plmalgpar, vecvar[2:end], vcat(1.,bold), J0, h0) #all rounds apart round 0

    return (beta=bnew, Jtensor=J, htensor=h, psl=pslike)


end

function plm_opt_old(data;
						min_separation::Int = 0,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        lambdaB::Real=0.0,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(2:data.T-1)),
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
						method_par::Symbol=:LD_LBFGS,
                        epsconvbeta::Real=1.0e-5,
						betatol::Real=1.0e-3,
                        maxitbeta::Int=10,
                        verbose::Bool=true,
						method_beta::Symbol=:LD_MMA,
						maxit_glob::Int = 100)

	if (data.T)-2 != length(bstart)
		throw(DimensionMismatch("Size of inverse temperature vector does not match number of rounds"))
	end

    counter::Int=0

	plmalgpar = PlmAlg(method_par,verbose,epsconv,maxit)

	plmalgbeta = PlmAlg(method_beta,verbose,epsconvbeta,betatol,maxitbeta)

	vecvar = set_vecvar(data,lambdaJ=lambdaJ,lambdaH=lambdaH,lambdaB=lambdaB,weight=weight)

	q=vecvar[1].q
	N=vecvar[1].N
	T = size(vecvar,1)

    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
	bnew = zeros(Float64,T-2)

	#init
	J = zeros(Float64, q, q, N, N)
	h = zeros(Float64, q, N)
	fail_p = zeros(Int, N)
	fail_b = zeros(Int, T-2)

    pslike = 0.0

	Jmat0, psl0 = MinimizePLAsym(plmalgpar, vecvar[1]) #round 0
	J0, h0 = FieldsMatrix2Tensor(Jmat0, N, q)

    println("Round 0 computation ended")

    while(any(Dbeta .> betatol) && counter < maxit_glob)

        Jtens, htens = Opt_Fields(plmalgpar, vecvar[2:end], vcat(1.,bold), J0, h0)

        bnew, pslike, fail_b = BetaMinimize(bold,plmalgbeta,vecvar[3:end],Jtens,htens,J0,h0)

        if 1 in fail_b
            error("Pseudo-likelihood minimization failed")
        end

        wisereldist!(Dbeta,bnew,bold)

        println(Dbeta)

        bold = copy(bnew)

        counter += 1

    end

    Jtens, htens = Opt_Fields(plmalgpar, vecvar[2:end], vcat(1.,bold), J0, h0)

    return (beta=bnew, Jtensor=Jtens, htensor=htens, psl=pslike)


end

############################################

function Opt_Fields(alg::PlmAlg,var::Vector{PlmVar}, beta::Vector{Float64},
	J0::Array{Float64,4}, h0::Array{Float64,2})

    @assert length(var) == length(beta)

    Jmat, vpsl, fail_p = ParMinimize(alg,var,beta,J0,h0)

    if 1 in fail_p
        error("Pseudo-likelihood minimization failed")
    end

    Jtens, htens = FieldsMatrix2Tensor(Jmat,var[1].N,var[1].q)

    return Jtens, htens

end

##################################


function ParMinimize(alg::PlmAlg,var::Vector{PlmVar}, beta::Vector{Float64},
	J0::Array{Float64,4}, h0::Array{Float64,2})

	@assert length(var) == length(beta)

    LL = (var[1].N - 1) * var[1].q2 + var[1].q
    x0 = zeros(Float64, LL)
    vecps = SharedArray{Float64}(var[1].N)
    fail = SharedArray{Int}(var[1].N)

    Jmat = @distributed hcat for site=1:var[1].N #1:12
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        # if alg.boolmask != nothing # constrain to zero boolmask variables
	    #     lb,ub=ComputeUL(alg,var,site,LL)
	    #     lower_bounds!(opt, lb)
	    #     upper_bounds!(opt, ub)
        # end
        min_objective!(opt, (x,g)->optimfunwrapper(x, g, site, var, beta, J0, h0))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
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

function ParMinimize(alg::PlmAlg,var::Vector{PlmVar}, beta::Vector{Float64},
	Jtemp::Matrix{Float64},J0::Array{Float64,4}, h0::Array{Float64,2})

	@assert length(var) == length(beta)

    LL = (var[1].N - 1) * var[1].q2 + var[1].q
    x0 = zeros(Float64,var[1].N,LL)
    vecps = SharedArray{Float64}(var[1].N)
    fail = SharedArray{Int}(var[1].N)

	@assert size(Jtemp,1) == LL

	x0 = copy(Jtemp)

    Jmat = @distributed hcat for site=1:var[1].N #1:12
        opt = Opt(alg.method, length(x0[:,site]))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        # if alg.boolmask != nothing # constrain to zero boolmask variables
	    #     lb,ub=ComputeUL(alg,var,site,LL)
	    #     lower_bounds!(opt, lb)
	    #     upper_bounds!(opt, ub)
        # end
        min_objective!(opt, (x,g)->optimfunwrapper(x, g, site, var, beta, J0, h0))
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


#Psl of a site and gradient of all times but zero employing its parameters
function PLsiteAndGrad!(vecJ::Vector{Float64}, grad::Vector{Float64},site::Int,
	plmvar::Vector{PlmVar},	beta::Vector{Float64},J0::Array{Float64,4},h0::Array{Float64,2})

    LL = length(vecJ)
    q2 = plmvar[1].q2
    q = plmvar[1].q
    gaugecol = plmvar[1].gaugecol
    N = plmvar[1].N
	T=size(plmvar,1)

    pseudolike = 0.0

    for i=1:LL-q
	    grad[i] = 2.0 * plmvar[1].lambdaJ * vecJ[i]
	end
	for i=(LL-q+1):LL
	    grad[i] = 4.0 * plmvar[1].lambdaH * vecJ[i]
	end

	vecene = zeros(Float64,q)
	expvecenesunorm = zeros(Float64,q)

    for t = 1:T

        M = plmvar[t].M
    	Z = sdata(plmvar[t].Z)
    	W = sdata(plmvar[t].W)

	   @inbounds for a = 1:M

 			fillvecene!(vecene,vecJ,site,a,plmvar[t],beta[t],J0,h0)
	        lnorm = log(sumexp(vecene))
	        expvecenesunorm .= exp.(vecene .- lnorm)
	        pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)
	        offset = 0
	        for i = 1:site-1
	            @simd for s = 1:q
	                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * beta[t] * expvecenesunorm[s]
	            end
	            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * beta[t]
	            offset += q2
	        end
            for i = site+1:N
	            @simd for s = 1:q
	                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * beta[t] * expvecenesunorm[s]
	            end
	            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * beta[t]
	            offset += q2
	        end
	        @simd for s = 1:q
	            grad[ offset + s ] += W[a] * beta[t] * expvecenesunorm[s]
	        end
			grad[ offset + Z[site,a] ] -= W[a] * beta[t]
	    end

	end
	pseudolike += L2norm_asym(vecJ, plmvar[1])
    return pseudolike
end


##############################

function BetaMinimize(alg::PlmAlg,var::Vector{PlmVar},beta_init::Vector{Float64},G::Array{Array{Float64,3}},E0::Array{Array{Float64,3}})

    if !( size(G,1) == size(E0,1) == length(var) )
        throw(DimensionMismatch("var, E0 and G number of rounds must coincides"))
    end

    T = size(G,1)
    pslike = zeros(Float64, T)
    x0vec = copy(beta_init)
    bnew = zeros(Float64, T)
    fail = zeros(Int, T)
    println(alg.maxit)

    for t=1:T
        x0 = [x0vec[t]]
        println(x0)
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
		xtol_rel!(opt, alg.xtol)
        maxeval!(opt, alg.maxit)
        lower_bounds!(opt, [0.0])
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,var[t],G[t],E0[t]))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("round = %d\t pl = %.4f\t time = %.4f\t", t+2, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        if ret==:FAILURE
            fail[t] = 1
        end
        pslike[t]=minf
        bnew[t] = minx[1]
    end
    return bnew, sum(pslike), fail

end


function PLbetaGrad!(beta::Vector{Float64},grad::Vector{Float64},var::PlmVar,
	G::Array{Float64,3},E0::Array{Float64,3})

    N = var.N
    q = var.q
    pseudolike = 0.0

    grad[1] = 2*var.lambdaB*beta[1]

    M = var.M
    Z = sdata(var.Z)
    W = sdata(var.W)

	for site=1:N
		for a=1:M
			lnorm = log(sumexp(beta[1]*G[site,a,:].+E0[site,a,:]))
			pseudolike -= W[a] * (beta[1]*G[site,a,Z[site,a]]+E0[site,a,Z[site,a]] - lnorm)
			grad[1] -= W[a]*( G[site,a,Z[site,a]] - sum( G[site,a,:] .* exp.(beta[1]*G[site,a,:].+E0[site,a,:] .- lnorm)) )
		end
    end

    pseudolike += var.lambdaB*beta[1]*beta[1]

    @printf("grad = %.4f\tpsl=%.4f\n",grad[1],pseudolike)

    return pseudolike

end

function BetaMinimize(beta::Vector{Float64},alg::PlmAlg,var::Vector{PlmVar},Jtens::Array{Float64,4},htens::Array{Float64,2},J0::Array{Float64,4},h0::Array{Float64,2})

    T=size(var,1)
    pslike=zeros(Float64,T)
    x0vec = copy(beta)
    bnew=zeros(Float64,T)
    fail = zeros(Int,T)
    println(alg.maxit)

    for t=1:T
        x0 = [x0vec[t]]
        println(x0)
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
		xtol_rel!(opt, alg.xtol)
        maxeval!(opt, alg.maxit)
        lower_bounds!(opt, [0.0])
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,var[t],Jtens,htens,J0,h0))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("round = %d\t pl = %.4f\t time = %.4f\t", t+2, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        if ret==:FAILURE
            fail[t] = 1
        end
        pslike[t]=minf
        bnew[t] = minx[1]
    end
    return bnew, sum(pslike), fail
end

function PLbetaGrad!(beta::Vector{Float64},grad::Vector{Float64},var::PlmVar,Jtens::Array{Float64,4},htens::Array{Float64,2},J0::Array{Float64,4},h0::Array{Float64,2})

    N = var.N
    q = var.q
    pseudolike = 0.0

    grad[1] = 2*var.lambdaB*beta[1]

    M = var.M
    Z = sdata(var.Z)
    W = sdata(var.W)

    bvecener = zeros(Float64,q)
    vecener = zeros(Float64,q)

	for site=1:N
		for a=1:M
			fillvecene!(bvecener,vecener,site,a,beta[1],var,Jtens,htens,J0,h0)
			lnorm = log(sumexp(bvecener))
			pseudolike -= W[a] * (bvecener[Z[site,a]] - lnorm)
			grad[1] -= W[a]*( vecener[Z[site,a]] - sum( vecener .* exp.(bvecener .- lnorm)) )
		end
    end

    pseudolike += var.lambdaB*beta[1]*beta[1]

    @printf("grad = %.4f\tpsl=%.4f\n",grad[1],pseudolike)

    return pseudolike

end


############################################

#Energy filling employing J0 and h0
function fillvecene!(vecene::Array{Float64,1},vecJ::Array{Float64,1},site::Int,a::Int,var::PlmVar,b::Float64,J0::Array{Float64,4},h0::Array{Float64,2})
    q=var.q
    N=var.N
    q2 = q*q
    Z = sdata(var.Z)

    @inbounds for l = 1:q
        offset::Int = 0
        scra::Float64 = 0.0
        for i = 1:site-1 # Begin sum_i \neq site J
            scra += b*vecJ[offset + l + q * (Z[i,a]-1)] + J0[l,Z[i,a],site,i]
            offset += q2
        end
        # skipping sum over residue site
    	for i = site+1:N
            scra +=  b*vecJ[offset + l + q * (Z[i,a]-1)] + J0[l,Z[i,a],site,i]
            offset += q2
        end # End sum_i \neq site J
        scra +=  b*vecJ[offset + l] + h0[l,site] # sum H
        vecene[l] = scra
    end
end

function fillvecene!(bvecener::Vector{Float64},vecener::Vector{Float64},site::Int,a::Int,beta::Float64,var::PlmVar,Jtens::Array{Float64,4},htens::Array{Float64,2},J0::Array{Float64,4},h0::Array{Float64,2})

    Z = sdata(var.Z)
    W = sdata(var.W)
    N = var.N
    q=var.q

    @inbounds for l = 1:q
        bener::Float64 = 0.0
        ener::Float64 = 0.0

        for i=1:site-1
            bener += beta*Jtens[l,Z[i,a],site,i]+J0[l,Z[i,a],site,i]
            ener += Jtens[l,Z[i,a],site,i]
        end

        for i=site+1:N
            bener += beta*Jtens[l,Z[i,a],site,i]+J0[l,Z[i,a],site,i]
            ener += Jtens[l,Z[i,a],site,i]
        end

        bener += beta*htens[l,site]+h0[l,site]
        ener += htens[l,site]

        bvecener[l] = bener
        vecener[l] = ener
    end

end

function fillvecene(J::Array{Float64,4},h::Array{Float64,2},N::Int,q::Int,site::Int,a::Int,Z::Array{Int64,2})

    vecener = zeros(Float64,q)

    for l=1:q
        for i=1:site-1
            vecener[l] += J[l,Z[i,a],site,i]
        end

        for i=site+1:N
            vecener[l] += J[l,Z[i,a],site,i]
        end

        vecener[l] += h[l,site]

    end

    return vecener

end

function EnerMatrix!(G::Array{Array{Float64,3}},E0::Array{Array{Float64,3}},var::Vector{PlmVar},
	J::Array{Float64,4},h::Array{Float64,2},J0::Array{Float64,4},h0::Array{Float64,2})

    T = size(var,1)
    N = var[1].N
    q = var[1].q
    q2=q*q

    for t=1:T
		Z = sdata(var[t].Z)
        for site=1:N
            for a=1:var[t].M
                G[t][site,a,:] = fillvecene(J,h,N,q,site,a,Z)
                E0[t][site,a,:] = fillvecene(J0,h0,N,q,site,a,Z)
            end
        end

    end

end
