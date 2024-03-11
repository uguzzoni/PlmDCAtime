function plmdcap0t(data;
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
                        gaugecol::Int=-1,
						weight::Symbol=:counts,
						beta::Vector{Float64}=Float64.(collect(2:data.T)),
						epsconv::Real=1.0e-25,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS)

    plmalg = PlmAlg(method,verbose,epsconv,maxit, boolmask)

    vecvar = set_vecvar(data,lambdaJ,lambdaH,gaugecol=gaugecol,weight=weight)

    J0, h0, psl0 = MinimizePLp0(plmalg,vecvar[1],1.0)

    println("Round 0 estimates done")

    Jmat, psl, fail = MinimizePLAsym(plmalg,vecvar,beta,J0,h0)

    if 1 in fail
        println("Pseudo-likelihood minimization failed")
    end

    score, FNAPC, Jtensor, htensor = ComputeScore(Jmat, vecvar, min_separation)

	return(PlmOut{4}(sdata(psl), Jtensor, htensor, score))

end

#Function which minimizes pseudo-likelihood of initial combinatorial library and returns related estimated parameters
function MinimizePLp0(alg::PlmAlg, var::PlmVar, beta::Float64)
    LL = (var.N - 1) * var.q2 + var.q
    x0 = zeros(Float64, LL)
    vecps = SharedArray{Float64}(var.N)

    Jmat = @distributed hcat for site=1:var.N #1:12
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        if alg.boolmask != nothing # constrain to zero boolmask variables
	        lb,ub=ComputeUL(alg,var,site,LL)
	        lower_bounds!(opt, lb)
	        upper_bounds!(opt, ub)
        end
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,site,var,beta))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("site = %d\t pl = %.4f\t time = %.4f\t", site, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        vecps[site] = minf
        minx
    end

    Jtens0, htens0 = TensCoupField(Jmat,var)
    return Jtens0, htens0, vecps

end

#Function which minimizes psl of all times but zero, employing initial energy parameters
function MinimizePLAsym(alg::PlmAlg, var::Vector{PlmVar}, beta::Vector{Float64}, J0::Array{Float64,4},h0::Array{Float64,2})

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

    return Jmat, vecps, fail
end

#Psl of a site and gradient for the initial library
function PLsiteAndGrad!(vecJ::Array{Float64,1}, grad::Array{Float64,1}, site::Int, plmvar::PlmVar, beta::Float64)
    LL = length(vecJ)
    q2 = plmvar.q2
    q = plmvar.q
    gaugecol = plmvar.gaugecol
    N = plmvar.N

    pseudolike = 0.0


    for i=1:LL-q
	    grad[i] = 2.0 * plmvar.lambdaJ * vecJ[i]
	end
	for i=(LL-q+1):LL
	    grad[i] = 4.0 * plmvar.lambdaH * vecJ[i]
	end

    vecene = zeros(Float64,q)
	expvecenesunorm = zeros(Float64,q)

    M = plmvar.M
    Z = sdata(plmvar.Z)
    W = sdata(plmvar.W)


    @inbounds for a = 1:M
        fillvecene!(vecene, vecJ,site,a, q, Z,N,beta)
        lnorm = log(sumexp(vecene))
        expvecenesunorm .= exp.(vecene .- lnorm)
        pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)
        offset = 0
        for i = 1:site-1
            @simd for s = 1:q
                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * beta * expvecenesunorm[s]
            end
            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * beta
            offset += q2
        end
        for i = site+1:N
            @simd for s = 1:q
                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * beta * expvecenesunorm[s]
            end
            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * beta
            offset += q2
        end
        @simd for s = 1:q
            grad[ offset + s ] += W[a] * beta * expvecenesunorm[s]
        end
        grad[ offset + Z[site,a] ] -= W[a] * beta
    end

    if 1 <= gaugecol <= q
        offset = 0;
        @inbounds for i=1:N-1
            for s=1:q
                grad[offset + gaugecol + q * (s - 1) ] = 0.0; # Gauge!!! set gradJ[a,q] = 0
                grad[offset + s + q * (gaugecol - 1) ] = 0.0; # Gauge!!! set gradJ[q,a] = 0
            end
            offset += q2
        end
        grad[offset + gaugecol] = 0.0 # Gauge!!! set gradH[q] = 0
    end

    pseudolike += L2norm_asym(vecJ, plmvar)
    return pseudolike

end

#Psl of a site and gradient of all times but zero employing its parameters
function PLsiteAndGrad!(vecJ::Vector{Float64}, grad::Vector{Float64},site::Int,plmvar::Vector{PlmVar},beta::Vector{Float64},J0::Array{Float64,4},h0::Array{Float64,2})


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


    for t = 1:T-1

        M = plmvar[t+1].M
    	Z = sdata(plmvar[t+1].Z)
    	W = sdata(plmvar[t+1].W)


	    @inbounds for a = 1:M
	        fillvecene!(vecene,vecJ,site,a,plmvar[t+1],beta[t],J0,h0)
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

	    if 1 <= gaugecol <= q
	        offset = 0;
	        @inbounds for i=1:N-1
	            for s=1:q
	                grad[offset + gaugecol + q * (s - 1) ] = 0.0; # Gauge!!! set gradJ[a,q] = 0
	                grad[offset + s + q * (gaugecol - 1) ] = 0.0; # Gauge!!! set gradJ[q,a] = 0
	            end
	            offset += q2
	        end
	        grad[offset + gaugecol] = 0.0 # Gauge!!! set gradH[q] = 0
	    end

	end
	pseudolike += L2norm_asym(vecJ, plmvar)
    return pseudolike
end

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

#New method of function L2norm_asym necessary for time zero estimates
function L2norm_asym(vec::Array{Float64,1}, plmvar::PlmVar)

    q = plmvar.q
    N = plmvar.N
    lambdaJ = plmvar.lambdaJ
    lambdaH = plmvar.lambdaH

    LL = length(vec)

    mysum1 = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1 += vec[i] * vec[i]
    end
    mysum1 *= lambdaJ

    mysum2 = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2 += vec[i] * vec[i]
    end
    mysum2 *= 2lambdaH

    return mysum1+mysum2
end
