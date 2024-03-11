#function from plmDCA



function MinimizePLAsym(alg::PlmAlg, var::PlmVar)

    LL = (var.N - 1) * var.q2 + var.q
    x0 = zeros(Float64, LL)
    vecps = SharedArray{Float64}(var.N)
    Jmat = @distributed hcat for site=1:var.N

        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        # if alg.boolmask != nothing # constrain to zero boolmask variables
        #     lb,ub=ComputeUL(alg,var,site,LL)
        #     lower_bounds!(opt, lb)
        #     upper_bounds!(opt, ub)
        # end
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,site,var))
        elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("site = %d\t pl = %.4f\t time = %.4f\t", site, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        vecps[site] = minf
        minx
    end
    return Jmat, vecps
end


function PLsiteAndGrad!(vecJ::Array{Float64,1},  grad::Array{Float64,1}, site::Int, plmvar::PlmVar)

    LL = length(vecJ)
    q2 = plmvar.q2
    q = plmvar.q
    gaugecol = plmvar.gaugecol
    N = plmvar.N
    M = plmvar.M
    Z = sdata(plmvar.Z)
    W = sdata(plmvar.W)


    for i=1:LL-q
        grad[i] = 2.0 * plmvar.lambdaJE * vecJ[i]
    end
    for i=(LL-q+1):LL
       grad[i] = 4.0 * plmvar.lambdaHE * vecJ[i]
    end

    vecene = zeros(Float64,q)
    expvecenesunorm = zeros(Float64,q)
    pseudolike = 0.0

    @inbounds for a = 1:M
        fillvecene!(vecene, vecJ,site,a, q, Z,N)
        lnorm = log(sumexp(vecene))
        expvecenesunorm .= exp.(vecene .- lnorm)
        pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)
        offset = 0
        for i = 1:site-1
            @simd for s = 1:q
                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] *  expvecenesunorm[s]
            end
            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a]
            offset += q2
        end
	for i = site+1:N
            @simd for s = 1:q
                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] *  expvecenesunorm[s]
            end
            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a]
            offset += q2
        end

        @simd for s = 1:q
            grad[ offset + s ] += W[a] *  expvecenesunorm[s]
        end
	grad[ offset + Z[site,a] ] -= W[a]
    end
	#
    # if 1 <= gaugecol <= q
    #     offset = 0;
    #     @inbounds for i=1:N-1
    #         for s=1:q
    #             grad[offset + gaugecol + q * (s - 1)  ] = 0.0; # Gauge!!! set gradJ[a,q] = 0
    #             grad[offset + s + q * (gaugecol - 1) ] = 0.0; # Gauge!!! set gradJ[q,a] = 0
    #         end
    #         offset += q2
    #     end
    #     grad[offset + gaugecol] = 0.0 # Gauge!!! set gradH[q] = 0
    # end
    pseudolike += L2norm_asym(vecJ, plmvar)

    return pseudolike
end


function fillvecene!(vecene::Array{Float64,1}, vecJ::Array{Float64,1},site::Int, a::Int, q::Int, sZ::DenseArray{Int,2},N::Int)

    q2 = q*q
    Z = sdata(sZ)

    @inbounds for l = 1:q
        offset::Int = 0
        scra::Float64 = 0.0
        for i = 1:site-1 # Begin sum_i \neq site J
            scra += vecJ[offset + l + q * (Z[i,a]-1)]
            offset += q2
        end
        # skipping sum over residue site
    	for i = site+1:N
            scra += vecJ[offset + l + q * (Z[i,a]-1)]
            offset += q2
        end # End sum_i \neq site J
        scra += vecJ[offset + l] # sum H
        vecene[l] = scra
    end
end

function L2norm_asym(vec::Array{Float64,1}, plmvar::PlmVar)
    q = plmvar.q
    N = plmvar.N
    lambdaJE = plmvar.lambdaJE
    lambdaHE = plmvar.lambdaHE

    LL = length(vec)

    mysum1 = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1 += vec[i] * vec[i]
    end
    mysum1 *= lambdaJE

    mysum2 = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2 += vec[i] * vec[i]
    end
    mysum2 *= 2lambdaHE

    return mysum1+mysum2
end
