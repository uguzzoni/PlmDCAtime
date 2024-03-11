#Looks for the optimal temperatures exploring a grid of values
function  plmdcagt(data, b2vec::Vector{Float64}, b3vec::Vector{Float64};

                        boolmask::Union{Array{Bool,2},Nothing}=nothing,
                        min_separation::Int = 0,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        gaugecol::Int=-1,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS)

    plmalg = PlmAlg(method, verbose, epsconv ,maxit, boolmask)


	vecvar = set_vecvar(data, lambdaJ, lambdaH, gaugecol=gaugecol, weight=weight)

    J0, h0, psl0 = MinimizePLp0(plmalg,vecvar[1],1.0)

    psl = beta_grid(plmalg, vecvar, b2vec, b3vec, J0, h0)

	#Jmat, pslike = MinimizePLAsym(plmalg, vecvar, beta)

	#score, FNAPC, Jtensor, htensor = ComputeScore(Jmat, vecvar, min_separation)

	#return(PlmOut{4}(sdata(pslike), Jtensor, htensor, score))

    return psl

end

function  plmdcagt(data, bedge::Matrix{Float64};

                        boolmask::Union{Array{Bool,2},Nothing}=nothing,
                        min_separation::Int = 0,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        gaugecol::Int=-1,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS)

    if (data.T)-2 != size(bedge,1)
        throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
    end

    plmalg = PlmAlg(method, verbose, epsconv ,maxit, boolmask)

	vecvar = set_vecvar(data, lambdaJ, lambdaH, gaugecol=gaugecol, weight=weight)

    J0, h0, psl0 = MinimizePLp0(plmalg,vecvar[1],1.0)

    psl = beta_grid(plmalg, vecvar, bedge, J0, h0)

    return psl

end

#function beta_grid(alg::PlmAlg, var::Vector{PlmVar}, b2vec::Vector{Float64}, b3vec::Vector{Float64}, J0::Array{Float64,4}, h0::Array{Float64,2})
#    psl=Matrix{Float64}(undef,length(b2vec)*length(b3vec),3)
#    b=Vector{Float64}(undef,3)
#    iter = collect.(Iterators.product(b2vec,b3vec))
#    counter = 0

#    for i in iter[:]
#        counter += 1
#        pushfirst!(i,1.)
#        Jmat, pslike,fail = MinimizePLAsym(alg, var, i, J0, h0)
#        if 1 in fail
#            psl[counter,:] = [i[2],i[3],NaN]
#        else
#            psl[counter,:] = [i[2],i[3],sum(pslike)]
#        end
#    end

#    return psl

#end

function beta_grid(alg::PlmAlg, var::Vector{PlmVar}, bedge::Matrix{Float64}, J0::Array{Float64,4}, h0::Array{Float64,2})

    T = size(var,1)
    bmat = Array{Array{Float64},1}(undef,T-2)

    for t=1:T-2
        bmat[t] = collect(bedge[t,1]:bedge[t,3]:bedge[t,2])
    end

    iter = collect.(Iterators.product(bmat...))

    psl=Matrix{Float64}(undef,length(iter),T-1)
    b=Vector{Float64}(undef,T-1)

    #psl = SharedMatrix{Float64}(length(iter),T-1)
    #b = SharedArray{Float64}(T-1)



    counter = 0

    #@distributed
    for i in iter[:]
        counter += 1
        #pushfirst!(i,1.)
        b = vcat(1.,i)
        Jmat, pslike,fail = MinimizePLAsym(alg, var, b, J0, h0)
        if 1 in fail
            psl[counter,:] = vcat(i,NaN)
        else
            psl[counter,:] = vcat(i,sum(pslike))
        end
    end

    return psl

end

function plm_opt(data;
	                     # decimation::Bool=false,
                        boolmask::Union{Array{Bool,2},Nothing}=nothing,
                        # fracmax::Real = 0.3,
                        # fracdec::Real = 0.1,
                        # remove_dups::Bool = true,
                        min_separation::Int = 0,
                        # max_gap_fraction::Real = 0.9,
                        # theta = :auto,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        lambdaB::Real=0.0,
                        gaugecol::Int=-1,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(3:data.T)),
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        epsconvbeta::Real=1.0e-15,
                        maxitbeta::Int=2,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS,
                        method_beta::Symbol=:LD_MMA)

    eps_glob::Float64 = 1.0e-4
    maxit_glob::Int = 100
    counter::Int=0

    plmalgpar = PlmAlg(method,verbose,epsconv,maxit,boolmask)

    plmalgbeta = PlmAlgBeta(method_beta,verbose,epsconvbeta,maxitbeta)

    vecvar = set_vecvar(data,lambdaJ,lambdaH,lambdaB,gaugecol=gaugecol,weight=weight)

    T = size(vecvar,1)
    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
    Jmat = zeros(Float64,vecvar[1].N,vecvar[1].N-1)
    fail_p = zeros(Int,vecvar[1].N)
    fail_b = zeros(Int,T-2)
    Jtens = zeros(Float64,vecvar[1].q,vecvar[1].q,vecvar[1].N,vecvar[1].N)
    htens = zeros(Float64,vecvar[1].q,vecvar[1].N)
    bnew = zeros(Float64,T-2)
    pslike = 0.0


    J0, h0, psl0 = MinimizePLp0(plmalgpar,vecvar[1],1.0)

    println("Round 0 computation ended")

    while(any(Dbeta .> eps_glob) && counter < maxit_glob)

        Jtens, htens = OptCoupFields(bold,plmalgpar,vecvar,J0,h0)

        bnew, pslike, fail_b = BetaMinimize(bold,plmalgbeta,vecvar,Jtens,htens,J0,h0)

        if 1 in fail_b
            error("Pseudo-likelihood minimization failed")
        end

        wisereldist!(Dbeta,bnew,bold)

        println(Dbeta)

        bold = copy(bnew)

        counter += 1

    end

    Jtens, htens = OptCoupFields(bold,plmalgpar,vecvar,J0,h0)

    return bnew, Jtens, htens, pslike


end

function plm_opt2(data;
	                     # decimation::Bool=false,
                        boolmask::Union{Array{Bool,2},Nothing}=nothing,
                        # fracmax::Real = 0.3,
                        # fracdec::Real = 0.1,
                        # remove_dups::Bool = true,
                        min_separation::Int = 0,
                        # max_gap_fraction::Real = 0.9,
                        # theta = :auto,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        lambdaB::Real=0.0,
                        gaugecol::Int=-1,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(3:data.T)),
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        epsconvbeta::Real=1.0e-15,
                        maxitbeta::Int=2,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS,
                        method_beta::Symbol=:LD_MMA)

    eps_shift::Float64 = 1.0e-3
    maxit_glob::Int = 100
    counter::Int=0

    plmalgpar = PlmAlg(method,verbose,epsconv,maxit,boolmask)

    plmalgbeta = PlmAlgBeta(method_beta,verbose,epsconvbeta,maxitbeta)

    vecvar = set_vecvar(data,lambdaJ,lambdaH,lambdaB,gaugecol=gaugecol,weight=weight)

    T = size(vecvar,1)
    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
    Jmat = zeros(Float64,vecvar[1].N,vecvar[1].N-1)
    fail_p = zeros(Int,vecvar[1].N)
    fail_b = zeros(Int,T-2)
    Jtens = zeros(Float64,vecvar[1].q,vecvar[1].q,vecvar[1].N,vecvar[1].N)
    htens = zeros(Float64,vecvar[1].q,vecvar[1].N)
    bnew = zeros(Float64,T-2)
    pslike = 0.0


    J0, h0, psl0 = MinimizePLp0(plmalgpar,vecvar[1],1.0)

    println("Round 0 computation ended")

    Jtens, htens = OptCoupFields(bold,plmalgpar,vecvar,J0,h0)

    while(any(Dbeta .> eps_shift) && counter < maxit_glob)

        bnew, pslike, fail_b = BetaMinimize(bold,plmalgbeta,vecvar,Jtens,htens,J0,h0)

        if 1 in fail_b
            error("Pseudo-likelihood minimization with respect to beta failed")
        end

        wisereldist!(Dbeta,bnew,bold)

        println(Dbeta)

        bold = copy(bnew)

        Jtens, htens = OptCoupFields(bold,plmalgpar,vecvar,J0,h0)

        counter += 1

    end

    plmalgb2 = PlmAlgBeta(:LD_MMA,verbose,epsconvbeta,10000)

    bnew, pslike, fail_b = BetaMinimize(bold,plmalgb2,vecvar,Jtens,htens,J0,h0)

    if all(fail_b .== 0)
        return bnew, Jtens, htens, pslike
    else
        error("Convergence not reached")
    end


end

function OptCoupFields(beta::Vector{Float64},alg::PlmAlg,var::Vector{PlmVar},J0::Array{Float64,4},h0::Array{Float64,2})

    bpar = vcat(1.,beta)

    Jmat, fail_p = ParMinimize(alg,var,bpar,J0,h0)

    if 1 in fail_p
        error("Pseudo-likelihood minimization failed")
    end

    Jtens, htens = TensCoupField(Jmat,var[1])

    return Jtens, htens

end

function plm_opt3(data;
	                     # decimation::Bool=false,
                        boolmask::Union{Array{Bool,2},Nothing}=nothing,
                        # fracmax::Real = 0.3,
                        # fracdec::Real = 0.1,
                        # remove_dups::Bool = true,
                        min_separation::Int = 0,
                        # max_gap_fraction::Real = 0.9,
                        # theta = :auto,
						lambdaJ::Real=0.01,
                        lambdaH::Real=0.01,
                        lambdaB::Real=0.0,
                        gaugecol::Int=-1,
						weight::Symbol=:counts,
						bstart::Vector{Float64}=Float64.(collect(3:data.T)),
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        epsconvbeta::Real=1.0e-15,
                        maxitbeta::Int=2,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS,
                        method_beta::Symbol=:LD_MMA)

    eps_glob::Float64 = 1.0e-4
    maxit_glob::Int = 100
    counter::Int=0

    plmalgpar = PlmAlg(method,verbose,epsconv,maxit,boolmask)

    plmalgbeta = PlmAlgBeta(method_beta,verbose,epsconvbeta,maxitbeta)

    vecvar = set_vecvar(data,lambdaJ,lambdaH,lambdaB,gaugecol=gaugecol,weight=weight)

    T = size(vecvar,1)
    Dbeta = ones(Float64,T-2)
    bold = copy(bstart)
    Jmat = zeros(Float64,vecvar[1].N,vecvar[1].N-1)
    fail_p = zeros(Int,vecvar[1].N)
    fail_b = zeros(Int,T-2)
    Jtens = zeros(Float64,vecvar[1].q,vecvar[1].q,vecvar[1].N,vecvar[1].N)
    htens = zeros(Float64,vecvar[1].q,vecvar[1].N)
    bnew = zeros(Float64,T-2)
    pslike = 0.0

    G = Array{Array{Float64,3}}(undef,T-2)
    E0 = Array{Array{Float64,3}}(undef,T-2)

    for t=1:T-2
        G[t] = zeros(Float64,vecvar[1].N,vecvar[t+2].M,vecvar[1].q)
        E0[t] = zeros(Float64,vecvar[1].N,vecvar[t+2].M,vecvar[1].q)
    end

    J0, h0, psl0 = MinimizePLp0(plmalgpar,vecvar[1],1.0)

    println("Round 0 computation ended")

    while(any(Dbeta .> eps_glob) && counter < maxit_glob)

        Jtens, htens = OptCoupFields(bold,plmalgpar,vecvar,J0,h0)

        EnerMatrix!(G,E0,vecvar,Jtens,htens,J0,h0)

        bnew, pslike, fail_b = BetaMinimize(bold,plmalgbeta,vecvar,G,E0)

        if 1 in fail_b
            error("Pseudo-likelihood minimization failed")
        end

        wisereldist!(Dbeta,bnew,bold)

        println(Dbeta)

        bold = copy(bnew)

        counter += 1

    end

    Jtens, htens = OptCoupFields(bold,plmalgpar,vecvar,J0,h0)

    return bnew, Jtens, htens, pslike


end

function EnerMatrix!(G::Array{Array{Float64,3}},E0::Array{Array{Float64,3}},var::Vector{PlmVar},J::Array{Float64,4},h::Array{Float64,2},J0::Array{Float64,4},h0::Array{Float64,2})

    T = size(var,1)
    N = var[1].N
    q = var[1].q
    q2=q*q
    Z = Array{Array{Int64,2}}(undef,T-2)

    for t=1:T-2
        Z[t] = sdata(var[t+2].Z)
    end

    for t=1:T-2
        for site=1:N
            for a=1:var[t+2].M
                G[t][site,a,:] = fillvecene(J,h,N,q,site,a,Z[t])
                E0[t][site,a,:] = fillvecene(J0,h0,N,q,site,a,Z[t])
            end
        end

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

function BetaMinimize(beta::Vector{Float64},alg::PlmAlgBeta,var::Vector{PlmVar},G::Array{Array{Float64,3}},E0::Array{Array{Float64,3}})

    if size(G,1) != size(E0,1)
        throw(DimensionMismatch("E0 and G number of rounds must coincides"))
    end

    T_eff = size(G,1)
    pslike = zeros(Float64,T_eff)
    x0vec = copy(beta)
    bnew = zeros(Float64,T_eff)
    fail = zeros(Int,T_eff)
    println(alg.maxitbeta)

    for t=1:T_eff
        x0 = [x0vec[t]]
        println(x0)
        opt = Opt(alg.method_beta, length(x0))
        ftol_abs!(opt, alg.epsconvbeta)
        maxeval!(opt, alg.maxitbeta)
        lower_bounds!(opt, [0.0])
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,var[t+2],G[t],E0[t]))
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

function PLbetaGrad!(beta::Vector{Float64},grad::Vector{Float64},var::PlmVar,G::Array{Float64,3},E0::Array{Float64,3})

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
			lnorm = log(sumexp(beta[1]*G[site,a,:].+E0[site,a,:]))
			pseudolike -= W[a] * (beta[1]*G[site,a,Z[site,a]]+E0[site,a,Z[site,a]] - lnorm)
			grad[1] -= W[a]*( G[site,a,Z[site,a]] - sum( G[site,a,:] .* exp.(beta[1]*G[site,a,:].+E0[site,a,:] .- lnorm)) )
		end
    end

    pseudolike += var.lambdaB*beta[1]*beta[1]

    @printf("grad = %.4f\tpsl=%.4f\n",grad[1],pseudolike)

    return pseudolike

end

function ParMinimize(alg::PlmAlg, var::Vector{PlmVar}, beta::Vector{Float64}, J0::Array{Float64,4}, h0::Array{Float64,2})

    LL = (var[1].N - 1) * var[1].q2 + var[1].q
    x0 = zeros(Float64, LL)
    vecps = SharedArray{Float64}(var[1].N)
    fail = SharedArray{Int}(var[1].N)

    Jmat = @distributed hcat for site=1:var[1].N #1:12
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        if alg.boolmask != nothing # constrain to zero boolmask variables
	        lb,ub=ComputeUL(alg,var,site,LL)
	        lower_bounds!(opt, lb)
	        upper_bounds!(opt, ub)
        end
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,site,var,beta,J0,h0))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("site = %d\t pl = %.4f\t time = %.4f\t", site, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        if ret==:FAILURE
            fail[site] = 1
        end
        vecps[site] = minf
        minx
    end
    return Jmat, fail
end

function BetaMinimize(beta::Vector{Float64},alg::PlmAlgBeta,var::Vector{PlmVar},Jtens::Array{Float64,4},htens::Array{Float64,2},J0::Array{Float64,4},h0::Array{Float64,2})

    T=size(var,1)
    pslike=zeros(Float64,T-2)
    x0vec = copy(beta)
    bnew=zeros(Float64,T-2)
    fail = zeros(Int,T-2)
    println(alg.maxitbeta)

    for t=1:T-2
        x0 = [x0vec[t]]
        println(x0)
        opt = Opt(alg.method_beta, length(x0))
        ftol_abs!(opt, alg.epsconvbeta)
        maxeval!(opt, alg.maxitbeta)
        lower_bounds!(opt, [0.0])
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,var[t+2],Jtens,htens,J0,h0))
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

function TensCoupField(Jmat::Matrix{Float64},var::PlmVar)
    q = var.q
    N = var.N

    JJ=reshape(Jmat[1:end-q,:], q,q,N-1,N)



    Jtemp1=zeros( q,q,Int(N*(N-1)/2))
    Jtemp2=zeros( q,q,Int(N*(N-1)/2))
    l = 1

    for i=1:(N-1)
        for j=(i+1):N
            Jtemp1[:,:,l]=JJ[:,:,j-1,i]; #J_ij as estimated from from g_i.
            Jtemp2[:,:,l]=JJ[:,:,i,j]; #J_ij as estimated from from g_j.
            l=l+1;
        end
    end


    Jtensor = zeros(q,q,N,N)
    l = 1
    for i = 1:N-1
        for j=i+1:N
            Jtensor[:,:,i,j] = Jtemp1[:,:,l]
            Jtensor[:,:,j,i] = Jtemp2[:,:,l]
            l += 1
        end
    end

    htensor = fill(0.0, q,N)
    for i in 1:N
        htensor[:,i] = Jmat[end-q+1:end,i]
    end

    return 0.5*(permutedims(Jtensor,[2,1,4,3])+Jtensor), htensor

end
