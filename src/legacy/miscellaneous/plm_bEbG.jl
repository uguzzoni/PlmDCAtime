# Computing ν*G+t*E parameters
#INPUT: data with T rounds and vectors of b_E=t and b_G = ν with T and elements (round 0 1 2 .. T-1)
#OUTPUT: G and E parameters (J and h)

function  plmdcat(data, bE::Vector{Float64}, bG::Vector{Float64};
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

	if (data.T) != length(bE) != length(bG)
		throw(DimensionMismatch("Number of temperature vectors does not match rounds"))
	end

	alg = PlmDCAt.PlmAlg(method, true, epsconv ,maxit)
	vecvar = PlmDCAt.set_vecvar(data,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJG=lambdaJG,lambdaHG=lambdaHG,weight=weight)

	q = vecvar[1].q
	N = vecvar[1].N
	q2 = vecvar[1].q2

	Jmat, pslike, fail = PlmDCAt.ParMinbEbG(alg,vecvar, bE, bG, Jstart);
	#pslvec = ComputePSL(Jmat,vecvar,beta);
	J_E,h_E,J_G,h_G = PlmDCAt.GEMat2TensNoGauge(Jmat,N,q);

	return (Jmat=Jmat, J_E=J_E, h_E=h_E, J_G=J_G, h_G=h_G, psl=pslike)#, pslv=pslvec)
end

#Function performing scan over β_E, multiplying G energy by β_G vector

function plmdcagt(vecvar::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, bG::Vector{Float64};
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

	psl = beta_grid(plmalg, vecvar, bgrid, Jstart, bG)

	return psl

end

#Function performing scan over β_E, multiplying G energy by β_G vector when round 0 is not available

function  plmdcagt_nr0(vecvar::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, bG::Vector{Float64};
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

    psl = bg_nr0(plmalg, vecvar, bgrid, Jstart, bG)

    return psl

end

#Computes beta grid over β_E taking β_G as fixed

function beta_grid(alg::PlmAlg, var::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, Jstart::Matrix{Float64}, bG::Vector{Float64})

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

    for i in iter[:]
        counter += 1
        b = vcat(0.,1.,i)
        Jmat, pslike, fail = ParMinbEbG(alg, var, b, bG, Jtemp)
        if 1 in fail
            psl[counter,:] = vcat(i,NaN)
        else
            psl[counter,:] = vcat(i,sum(pslike))
        end
		Jtemp = copy(Jmat)
    end

    return psl

end

#Computes beta grid over β_E taking β_G as fixed when round 0 is not available

function bg_nr0(alg::PlmAlg, var::Vector{PlmVar}, bgrid::Vector{Vector{Float64}}, Jstart::Matrix{Float64}, bG::Vector{Float64})

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
        Jmat, pslike, fail = ParMinbEbG(alg, var, b, bG, Jtemp)
        if 1 in fail
            psl[counter,:] = vcat(i,NaN)
        else
            psl[counter,:] = vcat(i,sum(pslike))
        end
		Jtemp = copy(Jmat)
    end

    return psl

end

#Functions computing optimal parameter values employing both fictious temperatures β_E and β_G

function ParMinbEbG(alg::PlmAlg, var::Vector{PlmVar}, bE::Vector{Float64}, bG::Vector{Float64},
	Jtemp::Matrix{Float64})

	@assert length(var) == length(bE) == length(bG)

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
        min_objective!(opt, (x,g)->optimfunwrapper(x, g, site, var, bE, bG))
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

function PLSiteGradbEbG!(vecJ::Vector{Float64}, grad::Vector{Float64},site::Int,
	plmvar::Vector{PlmVar},	bE::Vector{Float64}, bG::Vector{Float64})

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

 			fillvecene!(vecene,vecJ,site,a,plmvar[t],bE[t],bG[t])
	        lnorm = log(sumexp(vecene))
	        expvecenesumnorm .= exp.(vecene .- lnorm)
	        pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)
	        offset = 0
	        for i = 1:site-1
	            @simd for s = 1:q
	                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * bE[t] * expvecenesumnorm[s]
					grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] += W[a]* bG[t] * expvecenesumnorm[s]
					#grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] = 0.0
	            end
	            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * bE[t]
				grad[ LL + offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * bG[t]
				#grad[LL + offset + Z[site,a] + q * ( Z[i,a] - 1 )] = 0.0
	            offset += q2
	        end
            for i = site+1:N
	            @simd for s = 1:q
	                grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] * bE[t] * expvecenesumnorm[s]
					grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] += W[a] *bG[t]* expvecenesumnorm[s]
					#grad[ LL + offset + s + q * ( Z[i,a] - 1 ) ] = 0.0
	            end
	            grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * bE[t]
				grad[ LL + offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] * bG[t]
				#grad[LL + offset + Z[site,a] + q * ( Z[i,a] - 1 )] = 0.0
	            offset += q2
	        end
	        @simd for s = 1:q
	            grad[ offset + s ] += W[a] * bE[t] * expvecenesumnorm[s]
				grad[ LL + offset + s ] += W[a] * bG[t]* expvecenesumnorm[s]
	        end
			grad[ offset + Z[site,a] ] -= W[a] * bE[t]
			grad[ LL + offset + Z[site,a] ] -= W[a]*bG[t]
	    end

	end
	pseudolike += L2norm_asymEG(vecJ, plmvar[1])
    return pseudolike
end

function fillvecene!(vecene::Array{Float64,1},vecJ::Array{Float64,1},site::Int,a::Int,var::PlmVar,bE::Float64,bG::Float64)
	LL = Int(length(vecJ)/2)
	q=var.q
    N=var.N
    q2 = q*q
    Z = sdata(var.Z)

    @inbounds for l = 1:q
        offset::Int = 0
        scra::Float64 = 0.0
        for i = 1:site-1 # Begin sum_i \neq site j
            scra += bE*vecJ[offset + l + q * (Z[i,a]-1)] + bG*vecJ[LL + offset + l + q * (Z[i,a]-1)]
            offset += q2
        end
        # skipping sum over residue site
    	for i = site+1:N
			scra += bE*vecJ[offset + l + q * (Z[i,a]-1)] + bG*vecJ[LL + offset + l + q * (Z[i,a]-1)]
            offset += q2
        end # End sum_i \neq site j
        scra +=  bE*vecJ[offset + l] + bG*vecJ[LL+offset+l] # sum H
        vecene[l] = scra
    end
end
