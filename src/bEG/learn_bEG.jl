#Computing G+b*E parameters
#INPUT: data with T+1 rounds (including initial library) and vectors of beta with T+1 elements (round 0 1 2 .. T)
#OUTPUT: G and E parameters (J and h)

function  learn_par(data, beta::Vector{Float64};
						q::Int=20,
						N::Int=data.L,
						lambdaJE::Real = 0.01,
						lambdaHE::Real = 0.01,
						lambdaJG::Real = 0.01,
						lambdaHG::Real = 0.01,
						weight::Symbol = :counts,
						init::Symbol = :ParStart,
						#model::Symbol = :Epistasis,
						epsconv::Real = 1.0e-10,
						maxit::Int = 10000,
						verbose::Bool = true,
						algorithm::Symbol = :LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (data.T) != length(beta)
		throw(DimensionMismatch("beta elements does not match rounds"))
	end

	w = Workspace(data, weight=weight, lambdaJE=lambdaJE,lambdaHE=lambdaHE, lambdaJG=lambdaJG,lambdaHG=lambdaHG, algorithm = algorithm, verbose = verbose, epsconv = epsconv, maxit = maxit)

	#learn model
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)
	x, pslike, fail = optimize(x, w, beta; w.opt_args...)

	JE,hE,JG,hG = Par2Tensor(x, w.q, w.N)

	return (Jmat=x, JE=JE, hE = hE , JG = JG, hG = hG, psl=pslike)
end

# #Computing G+b*E parameters
# #INPUT: workspace variable with T+1 rounds (including initial library) and vectors of beta with T+1 elements (round 0 1 2 .. T)
# #OUTPUT: G and E parameters (J and h)

function  learn_par(w::Workspace, beta::Vector{Float64};
						q::Int=20,
						N::Int=w.N,
						init::Symbol = :ParStart,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (w.T) != length(beta)
		throw(DimensionMismatch("beta elements does not match rounds"))
	end

	#learn model
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)
	x, pslike, fail = optimize(x, w, beta; w.opt_args...)

	JE,hE,JG,hG = Par2Tensor(x, w.q, w.N)

	return (Jmat=x, JE=JE, hE = hE , JG = JG, hG = hG, psl=pslike)
end


"""
 Computing H = G + β(t) * E parameters for a grid of beta values

 INPUT: data with T samples and for each beta element a range of values
 OUTPUT: pseudolikelihood values for each grid point
"""

function  learn_par(data, betagrid::Vector{Vector{Float64}};
						q::Int=20,
						N::Int=data.L,
						lambdaJE::Real = 0.01,
						lambdaHE::Real = 0.01,
						lambdaJG::Real = 0.01,
						lambdaHG::Real = 0.01,
						weight::Symbol = :counts,
						init::Symbol = :ParStart,
						#model::Symbol = :Epistasis,
						epsconv::Real = 1.0e-10,
						maxit::Int = 10000,
						verbose::Bool = true,
						algorithm::Symbol = :LD_LBFGS,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (data.T-2) != length(betagrid)
		throw(DimensionMismatch("beta elements does not match (number rounds - 2)"))
	end


	w = Workspace(data, weight=weight, lambdaJE=lambdaJE,lambdaHE=lambdaHE, lambdaJG=lambdaJG,lambdaHG=lambdaHG, algorithm = algorithm, verbose = verbose, epsconv = epsconv, maxit = maxit)


	#learn model for each beta grid values
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)

    iter = collect.(Iterators.product(betagrid...))
    psl = Vector{Float64}(undef,length(iter))
    b = Vector{Float64}(undef,data.T)

    counter = 0
    for i in iter
        counter += 1
		b = vcat(0.,1.,i)
		x, pslike, fail = optimize(x, w, b; w.opt_args...)
        if 1 in fail
            psl[counter] = NaN
        else
            psl[counter] = sum(pslike)
        end
    end

    return psl
end

function  learn_par(w::Workspace, betagrid::Vector{Vector{Float64}};
						q::Int=20,
						N::Int=w.N,
						init::Symbol = :ParStart,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (w.T-2) != length(betagrid)
		throw(DimensionMismatch("beta elements does not match (number rounds - 2)"))
	end

	#learn model for each beta grid values
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)

	iter = collect.(Iterators.product(betagrid...))
	psl = 0.0
	b = Vector{Float64}(undef,w.T)
	tab = Matrix{Float64}(undef,length(iter),w.T-1)

	counter = 0
	for i in iter
		counter += 1
		b = vcat(0.,1.,i)
		x, pslike, fail = optimize(x, w, b; w.opt_args...)
		if 1 in fail
			psl = NaN
		else
			psl = sum(pslike)
		end
		tab[counter,:] = vcat(i,psl)
	end

	return tab
end


"""
 Computing optimal H' = H0 + beta_t * H and beta parameters

 INPUT: data with T samples
 OUTPUT: H0 and H parameters (J and h) and beta
"""

function  learn_par(data;
						q::Int=20,
						N::Int=data.L,
						bstart::Vector{Float64} = collect(2.:1.:(data.T-1)),
						lambdaJE::Real = 0.01,
						lambdaHE::Real = 0.01,
						lambdaJG::Real = 0.01,
						lambdaHG::Real = 0.01,
						weight::Symbol = :counts,
						init::Symbol = :ParStart,
						epsconv::Real = 1.0e-10,
						maxit::Int = 10000,
						verbose::Bool = true,
						algorithm::Symbol = :LD_LBFGS,
						betatol::Real = 1.e-3,
						maxit_global::Int = 100,
						b_iter::Int = 1,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (data.T - 2) != length(bstart)
	 	throw(DimensionMismatch("betastart elements does not match (number rounds - 2)"))
	end

	w = Workspace(data, weight=weight, lambdaJE=lambdaJE, lambdaHE=lambdaHE, lambdaJG=lambdaJG, lambdaHG=lambdaHG, algorithm = algorithm, verbose = verbose, epsconv = epsconv, maxit = maxit)

	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)
	beta = copy(bstart)
	Dbeta = ones(Float64,w.T-2)
	fail_p = zeros(Int, w.N)
	counter = 0

	x, pslike, fail = optimize(x, w, vcat(0.0, 1.0, beta); w.opt_args...)
	println("beta start: ",beta)

	while( any(Dbeta .> betatol) && counter < maxit_global )

			bold=copy(beta);
			BetaOptimize!(beta, x, w, verbose = w.opt_args.verbose, ftol = betatol)
			Dbeta = abs.(bold.-beta)./beta
			println("> ",beta," (",Dbeta,")\n")

			x, pslike, fail = optimize(x, w, vcat(0.0, 1.0, beta); w.opt_args...)

			if 1 in fail_p
				error("Pseudo-likelihood minimization failed")
			end
	        counter += 1
	end

	JE,hE,JG,hG = Par2Tensor(x, w.q, w.N)

	return (beta=beta, JE=JE, hE=hE, JG=JG, hG=hG, psl=pslike)

end
