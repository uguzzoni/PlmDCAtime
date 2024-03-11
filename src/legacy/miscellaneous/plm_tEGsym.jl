function  plmdcat_sym(data, beta::Vector{Float64};
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
						Jstart::Vector{Float64}=zeros(Float64,Int(2*(N*(N-1)/2*q*q+N*q))))

	if (data.T) != length(beta)
		throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, true, epsconv ,maxit)
	vecvar = PlmDCAt.set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

	Jvec, pslike=PlmDCAt.MinimizePLSym(alg,vecvar,beta,Jstart);
	J_E,h_E,J_G,h_G = PlmDCAt.EGVec2TensSym(Jvec,vecvar[1]);

	return (J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike, Jvec=Jvec)
end

function MinimizePLSym(alg::PlmAlg, var::Vector{PlmVar}, beta::Vector{Float64}, Jstart::Vector{Float64})

    N  = var[1].N
    q  = var[1].q
    q2 = var[1].q2

    Nc2 = binomial(N,2)
    LL  = Nc2 * q2  + N * q

	@assert length(Jstart)==2*LL

    x0 = copy(Jstart)

    opt = Opt(alg.method, length(x0))

    ftol_abs!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    min_objective!(opt, (x,g)->optimfunwrapper(x,g,var,beta))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t exit status = ", minf, elapstime)
    alg.verbose && println(ret)

    return minx, minf
end

function PLsiteAndGradSym!(vecJ::Vector{Float64}, grad::Vector{Float64}, var::Vector{PlmVar}, beta::Vector{Float64})

	LL = Int(length(vecJ)/2)
	q2 = var[1].q2
	q = var[1].q
	N = var[1].N

	T = size(var,1)

	@assert LL == N*(N-1)/2*q2+N*q

	for i=1:LL-N*q
		grad[i] = 2.0 * vecJ[i] * var[1].lambdaJE
		grad[LL+i] = 2.0 * vecJ[LL+i] * var[1].lambdaJG
	end

	for i=(LL-N*q + 1):LL
		grad[i] = 2.0 * vecJ[i] * var[1].lambdaHE
		grad[LL+i] = 2.0 * vecJ[LL+i] * var[1].lambdaHG
	end

	pseudolike = 0.0

	for t=1:T
		M = var[t].M
		Z=var[t].Z
		W=var[t].W
		for a = 1:M
			pseudolike += ComputePatternPLSym!(grad, vecJ, Z[:,a], W[a], N, q, q2, beta[t])
		end
	end

	pseudolike += L2norm_sym(vecJ, var[1])
	return pseudolike

end

function ComputePatternPLSym!(grad::Array{Float64,1}, vecJ::Array{Float64,1}, Z::Array{Int,1}, Wa::Float64, N::Int, q::Int, q2::Int, beta::Float64)

	LL = Int(length(vecJ)/2)
    vecene = zeros(Float64,q)
    expvecenesunorm = zeros(Float64,q)
    pseudolike = 0.0
    offset = mygetindex(N-1, N, q, q, N, q, q2)
    @inbounds for site=1:N    # site < i
        fillvecenesym!(vecene, vecJ, Z, site, q, N, beta)
        lnorm = log(sumexp(vecene))
        expvecenesunorm = exp.(vecene .- lnorm)
        pseudolike -= Wa * ( vecene[Z[site]] - lnorm )
		for i = 1:(site-1)
            for s = 1:q
                grad[ mygetindex(i, site, Z[i], s, N, q, q2) ] += 0.5 * beta * Wa * expvecenesunorm[s]
				grad[ LL + mygetindex(i, site, Z[i], s, N, q, q2) ] += 0.5 * Wa * expvecenesunorm[s]
            end
            grad[ mygetindex(i, site , Z[i], Z[site],  N,q,q2) ] -= 0.5 *beta* Wa
			grad[ LL + mygetindex(i, site , Z[i], Z[site],  N,q,q2) ] -= 0.5 * Wa

    	end
		for i = (site+1):N
            for s = 1:q
                grad[ mygetindex(site, i, s,  Z[i], N,q,q2) ] += 0.5 * beta * Wa * expvecenesunorm[s]
				grad[ LL + mygetindex(site, i, s,  Z[i], N,q,q2) ] += 0.5 * Wa * expvecenesunorm[s]
            end
            grad[ mygetindex(site, i, Z[site], Z[i], N,q,q2) ] -= 0.5 * beta * Wa
			grad[ LL + mygetindex(site, i, Z[site], Z[i], N,q,q2) ] -= 0.5 * Wa
    	end
    	@simd for s = 1:q
            grad[ offset + s ] += Wa * beta * expvecenesunorm[s]
			grad[ offset + LL + s ] += Wa * expvecenesunorm[s]
    	end
		grad[ offset + Z[site] ] -= Wa * beta
		grad[ offset + LL + Z[site] ] -= Wa
 		offset += q
    end
    return pseudolike
end

function fillvecenesym!(vecene::Array{Float64,1}, vecJ::Array{Float64,1}, Z::Array{Int64,1}, site::Int, q::Int ,N::Int, beta::Float64)
    q2 = q*q
	LL = Int(N*(N-1)/2*q2+N*q)
    @inbounds begin
        for l = 1:q
            offset::Int = 0
            scra::Float64 = 0.0

            for i=1:site-1
                scra += beta*vecJ[ mygetindex(i, site, Z[i], l,  N, q, q2)] + vecJ[ LL + mygetindex(i, site, Z[i], l,  N, q, q2)]
            end
    	    for i = site+1:N
                scra += beta*vecJ[ mygetindex(site, i, l, Z[i], N, q, q2)] + vecJ[ LL + mygetindex(site, i, l, Z[i], N, q, q2)]
            end # End sum_i \neq site J
            offset = mygetindex(N-1, N, q, q, N, q, q2)  + ( site - 1) * q  # last J element + (site-1)*q
            scra += beta*vecJ[offset + l] + vecJ[offset + LL + l] # sum H
            vecene[l] = scra
        end
    end
end

function L2norm_sym(vec::Array{Float64,1}, var::PlmVar)

    q = var.q
    N = var.N

    LL = Int(length(vec)/2)


    mysum1E = 0.0
	mysum1G = 0.0
    @inbounds @simd for i=1:(LL-N*q)
        mysum1E += vec[i] * vec[i]
		mysum1G += vec[i+LL]*vec[i+LL]
    end
    mysum1E *= var.lambdaJE
	mysum1G *= var.lambdaJG

    mysum2E = 0.0
	mysum2G = 0.0
    @inbounds @simd for i=(LL - N*q + 1):LL
        mysum2E += vec[i] * vec[i]
		mysum2G += vec[i+LL]*vec[i+LL]
    end
    mysum2E *= var.lambdaHE
	mysum2G *= var.lambdaHG


    return mysum1E+mysum1G+mysum2E+mysum2G
end


function mygetindex( i::Int, j::Int, coli::Int, colj::Int, N::Int, q::Int, q2::Int)
    offset_i = ( (i-1) * N  - ( (i * ( i -1 ) ) >> 1 ) ) * q2 # (i-1) N q2 + i (i-1) q2 / 2
    offset_j = (j - i - 1 ) * q2
    return offset_i + offset_j + coli + q * (colj - 1)
end

function EGVec2TensSym(Jvec::Array{Float64,1}, var::PlmVar)

    LL = Int(length(Jvec)/2)
    N=var.N
    q=var.q
    Nc2 = binomial(N,2)

	@assert LL == Nc2*q*q+N*q

    JEtens=reshape(Jvec[1:LL-N*q],q,q,Nc2)
    hEtens=fill(0.0,q,N)
	JGtens=reshape(Jvec[LL+1:2*LL-N*q],q,q,Nc2)
	hGtens=fill(0.0,q,N)

    hEtens=reshape(Jvec[LL-N*q + 1:LL],q,N)
    hGtens=reshape(Jvec[2*LL-N*q + 1:end],q,N)

    return inflate_matrix(JEtens,N),hEtens,inflate_matrix(JGtens,N),hGtens
end

function  plmdcagt_sym(data, bgrid::Vector{Vector{Float64}};
						lambdaJE::Real=0.01,
                        lambdaHE::Real=0.01,
						lambdaJG::Real=0.01,
						lambdaHG::Real=0.01,
						weight::Symbol=:counts,
						epsconv::Real=1.0e-5,
                        maxit::Int=10000,
                        verbose::Bool=true,
                        method::Symbol=:LD_LBFGS)

    if (data.T)-2 != length(bgrid)
        throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
    end

    plmalg = PlmAlg(method, verbose, epsconv ,maxit)

	vecvar = set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

    psl = beta_grid_sym(plmalg, vecvar, bgrid)

    return psl

end

function beta_grid_sym(alg::PlmAlg, var::Vector{PlmVar}, bgrid::Vector{Vector{Float64}})

    T = size(var,1)
	q = var[1].q
	N = var[1].N
	q2 = var[1].q2
	Nc2 = binomial(N,2)
	Jtemp = zeros(Float64,2*(Nc2*q2+N*q))
	Jmat = zeros(Float64,2*(Nc2*q2+N*q))

	@assert (T-2) == length(bgrid)


    iter = collect.(Iterators.product(bgrid...))

    psl=Matrix{Float64}(undef,length(iter),T-1)
    b=Vector{Float64}(undef,T)

    counter = 0

    for i in iter[:]
        counter += 1
        b = vcat(0.,1.,i)
        Jmat, pslike = MinimizePLSym(alg, var, b, Jtemp)
        psl[counter,:] = vcat(i,sum(pslike))
		Jtemp = copy(Jmat)
    end

    return psl

end
