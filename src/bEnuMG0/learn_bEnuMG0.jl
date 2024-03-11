#Computing G+ν(t)*M+β(t)*E parameters
#INPUT: data with T+1 rounds (including initial library) and vectors of beta and nu with T+1 elements (round 0 1 2 .. T)
#OUTPUT: G, M and E parameters (J and h)

# function  learn_par(data, beta::Vector{Float64}, nu::Vector{Float64};
# 						q::Int=20,
# 						N::Int=data.L,
# 						lambdaJE::Real = 0.01,
# 						lambdaHE::Real = 0.01,
# 						lambdaJM::Real=0.01,
# 						lambdaHM::Real=0.01,
# 						lambdaJG::Real = 0.01,
# 						lambdaHG::Real = 0.01,
# 						weight::Symbol = :counts,
# 						init::Symbol = :ParStart,
# 						#model::Symbol = :Epistasis,
# 						epsconv::Real = 1.0e-10,
# 						maxit::Int = 10000,
# 						verbose::Bool = true,
# 						algorithm::Symbol = :LD_LBFGS,
# 						ParStart::Matrix{Float64}=zeros(Float64,3*((N-1)*q*q+q),N))

# 	if (data.T) != length(beta)
# 		throw(DimensionMismatch("beta elements does not match rounds"))
# 	end

# 	if (data.T) != length(nu)
# 		throw(DimensionMismatch("nu elements does not match rounds"))
# 	end

# 	w = Workspace(data,weight=weight,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJM=lambdaJM,lambdaHM=lambdaHM,lambdaJG=lambdaJG,lambdaHG=lambdaHG,algorithm=algorithm,verbose=verbose,epsconv=epsconv,maxit=maxit)

# 	#learn model
# 	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, ParStart)
# 	x, pslike, fail = optimize(x, w, beta, nu; w.opt_args...)

# 	JE,hE,JM,hM,JG,hG = Par2Tensor(x, w.q, w.N)

# 	return (ParMat=x, JE=JE, hE=hE, JM=JM, hM=hM, JG=JG, hG=hG, psl=pslike)
# end

#Computing G over round 0 only, E and M over subsequent rounds
#INPUT: data with T+1 rounds (including initial library) and vectors of beta and nu with T elements (round 1 2 .. T)
#OUTPUT: G, M and E parameters (J and h)

function  learn_par(w::Workspace, beta::Vector{Float64}, nu::Vector{Float64};
						q::Int=20,
						N::Int=w.N,
						init::Symbol = :ParStart,
						ParStart0::Matrix{Float64}=zeros(Float64,(N-1)*q*q+q,N),
						ParStart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (w.T-1) != length(beta)
		throw(DimensionMismatch("beta elements does not match rounds"))
	end

	if (w.T-1) != length(nu)
		throw(DimensionMismatch("nu elements does not match rounds"))
	end

	#learn model
	x0 = initialization(init, w.samples[1], w.LL, w.q, w.N, ParStart0)
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, ParStart)
	x0, ps0like, fail0 = optimize(x0, w; w.opt_args...)
	x, pslike, fail = optimize(x, w, beta, nu, x0; w.opt_args...)

	JG,hG = Par2Tensor(x0, w.q, w.N)
	JE,hE = Par2Tensor(x[1:w.LL,:], w.q, w.N)
	JM,hM = Par2Tensor(x[w.LL+1:end,:], w.q, w.N)

	return (ParMat0=x0, ParMat=x, JE=JE, hE=hE, JM=JM, hM=hM, JG=JG, hG=hG, psl=pslike)
end

"""
 Computing H = G + ν(t) * M + β(t) * E parameters for a grid of beta and nu values

 INPUT: data with T samples and for each beta and nu element a range of values
 OUTPUT: pseudolikelihood values for each grid point
"""

# function  learn_par(data, betagrid::Vector{Vector{Float64}}, nugrid::Vector{Vector{Float64}};
# 						q::Int=20,
# 						N::Int=data.L,
# 						lambdaJE::Real = 0.01,
# 						lambdaHE::Real = 0.01,
# 						lambdaJM::Real=0.01,
# 						lambdaHM::Real=0.01,
# 						lambdaJG::Real = 0.01,
# 						lambdaHG::Real = 0.01,
# 						weight::Symbol = :counts,
# 						init::Symbol = :ParStart,
# 						epsconv::Real = 1.0e-10,
# 						maxit::Int = 10000,
# 						verbose::Bool = true,
# 						algorithm::Symbol = :LD_LBFGS,
# 						Jstart::Matrix{Float64}=zeros(Float64,3*((N-1)*q*q+q),N))

# 	if (data.T-2) != length(betagrid)
# 		throw(DimensionMismatch("beta elements does not match (number rounds - 2)"))
# 	end

# 	if (data.T-2) != length(nugrid)
# 		throw(DimensionMismatch("nu elements does not match (number rounds - 2)"))
# 	end


# 	w = Workspace(data,weight=weight,lambdaJE=lambdaJE,lambdaHE=lambdaHE,lambdaJM=lambdaJM,lambdaHM=lambdaHM,lambdaJG=lambdaJG,lambdaHG=lambdaHG,algorithm=algorithm,verbose=verbose,epsconv=epsconv,maxit=maxit)


# 	#learn model for each beta grid values
# 	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)

# 	iter_b = collect.(Iterators.product(betagrid...))
# 	iter_nu = collect.(Iterators.product(nugrid...))
#     psl = 0.0
# 	b = Vector{Float64}(undef,w.T)
# 	tab = Matrix{Float64}(undef,2*w.T-3,length(iter_b)*length(iter_nu))

# 	counter = 0
# 	for i in iter_b
# 		for j in iter_nu
# 			counter += 1
# 			b = vcat(0.,1.,i)
# 			nu = vcat(0.,1.,j)
# 			x, pslike, fail = optimize(x, w, b, nu; w.opt_args...)
# 			if 1 in fail
# 				psl = NaN
# 			else
# 				psl = sum(pslike)
# 			end
# 			tab[:,counter] = vcat(i,j,psl)
# 		end

# 	end

# 	return tab
# end

# function  learn_par(w::Workspace, betagrid::Vector{Vector{Float64}}, nugrid::Vector{Vector{Float64}};
# 						q::Int=20,
# 						N::Int=w.N,
# 						init::Symbol = :ParStart,
# 						ParStart::Matrix{Float64}=zeros(Float64,3*((N-1)*q*q+q),N))

# 	if (w.T-2) != length(betagrid)
# 		throw(DimensionMismatch("beta elements does not match (number rounds - 2)"))
# 	end

# 	if (w.T-2) != length(nugrid)
# 		throw(DimensionMismatch("nu elements does not match (number rounds - 2)"))
# 	end	


# 	#learn model for each beta grid values
# 	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, ParStart)

# 	iter_b = collect.(Iterators.product(betagrid...))
# 	iter_nu = collect.(Iterators.product(nugrid...))
# 	psl = 0.0
# 	b = Vector{Float64}(undef,w.T)
# 	tab = Matrix{Float64}(undef,length(iter_b)*length(iter_nu),2*w.T-3)

# 	counter = 0
# 	for i in iter_b
# 		for j in iter_nu
# 			counter += 1
# 			b = vcat(0.,1.,i)
# 			nu = vcat(0.,1.,j)
# 			x, pslike, fail = optimize(x, w, b, nu; w.opt_args...)
# 			if 1 in fail
# 				psl = NaN
# 			else
# 				psl = sum(pslike)
# 			end
# 			tab[counter,:] = vcat(i,j,psl)
# 		end

# 	end

# 	return tab
# end

# #Performing grid over the possible regularization multipliers returning Pearson correlation
# #between true and inferred energies

# function  learn_par(TEner::Vector{Float64}, seq::Array{Array{Int,1}}, b::Vector{Float64}, nu::Vector{Float64}, samp::Vector{DataSample}, λEv::Vector{Float64}, λMv::Vector{Float64}, λGv::Vector{Float64};
# 						q::Int=20,
# 						N::Int=size(samp[1].Z,1),
# 						init::Symbol = :ParStart,
# 						ParStart::Matrix{Float64}=zeros(Float64,3*((N-1)*q*q+q),N))
# 	pc = 0.0
# 	tab = Matrix{Float64}(undef,length(λEv)*length(λMv)*length(λGv),4)
	

# 	counter = 0
# 	for i in λEv
# 		for j in λMv
# 			for k in λGv
# 				counter += 1
# 				w = Workspace(samp;lambdaJE=i,lambdaHE=i,lambdaJM=j,lambdaHM=j,lambdaJG=k,lambdaHG=k)
# 				x = initialization(init,[w.samples[1],w.samples[w.T]],w.LL,w.q,w.N,ParStart)
# 				x, pslike, fail = optimize(x, w, b, nu; w.opt_args...)
# 				JE,hE,JM,HM,JG,hG = Par2Tensor(x, w.q, w.N)
# 				if 1 in fail
# 					error("Optimization failed")
# 				end
# 				E_inf = [compute_energy_seq(JE,hE,[s...]) for s in seq]
# 				pc = cor(E_inf,TEner)
# 				tab[counter,:] = vcat(i,j,k,pc)
# 			end
# 		end
# 	end

# 	return tab
# end


"""
 Computing optimal H' = H0 + beta_t * H and beta parameters

 INPUT: data with T samples
 OUTPUT: H0 and H parameters (J and h) and beta
"""

# function  learn_par(data;
# 						q::Int=20,
# 						N::Int=data.L,
# 						betastart::Vector{Float64} = collect(2.:1.:(data.T-1)),
# 						lambdaJE::Real = 0.01,
# 						lambdaHE::Real = 0.01,
# 						lambdaJG::Real = 0.01,
# 						lambdaHG::Real = 0.01,
# 						weight::Symbol = :counts,
# 						init::Symbol = :ParStart,
# 						#model::Symbol = :Epistasis,
# 						epsconv::Real = 1.0e-6,
# 						maxit::Int = 10000,
# 						verbose::Bool = true,
# 						algorithm::Symbol = :LD_LBFGS,
# 						betatol::Real = 1.e-3,
# 						maxit_global::Int = 100,
# 						b_iter::Int = 1,
# 						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

# 	if (data.T - 2) != length(betastart)
# 	 	throw(DimensionMismatch("betastart elements does not match (number rounds - 2)"))
# 	end

# 	w = Workspace(data, weight=weight, lambdaJE=lambdaJE, lambdaHE=lambdaHE, lambdaJG=lambdaJG, lambdaHG=lambdaHG, algorithm = algorithm, verbose = verbose, epsconv = epsconv, maxit = maxit)

# 	x = initialization(init, w.samples[w.T], w.LL, w.q, w.N)
# 	beta = copy(betastart)
# 	Dbeta = ones(Float64,w.T-2)
# 	fail_p = zeros(Int, w.N)
# 	counter = 0

# 	x, pslike, fail = optimize(x, w, vcat(0.0, 1.0, beta), w.opt_args...)
# 	println("beta start: ",beta)

# 	while( any(Dbeta .> betatol) && counter < maxit_global )

# 			bold=copy(beta);
# 			BetaOptimize!(beta, x, w, verbose = w.opt_args.verbose, ftol = betatol)
# 			Dbeta = abs.(bold.-beta)./beta
# 			println("> ",beta," (",Dbeta,")\n")

# 			x, pslike, fail = optimize(x, w, vcat(0.0, 1.0, beta), w.opt_args...)

# 			if 1 in fail_p
# 				error("Pseudo-likelihood minimization failed")
# 			end
# 	        counter += 1
# 	end

# 	JE,hE,JG,HG = Par2Tensor(x, w.q, w.N)

# 	return (beta=beta, JE=JE, hE=hE, JG=JG, hG=hG, psl=pslike)

# end
