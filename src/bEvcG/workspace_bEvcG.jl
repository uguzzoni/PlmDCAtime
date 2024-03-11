export Workspace, update_workspace!




#### !!!! change workspace -> w0, w,
##### field regularizzation -> free number of parameters (depends on models)

"""
    round data variable

Stores data and metaparameters.
"""

struct DataSample

    Z:: SharedArray{Int,2}
    W::SharedArray{Float64,1}
    M::Int
    IdxZ::SharedArray{Int,2} #partial index computation for speed up energy calculation

    function DataSample(Z,W,M,IdxZ)
        sZ = SharedArray{Int}(size(Z))
        sZ[:] = Z
        sW = SharedArray{Float64}(size(W))
        sW[:] = W
        sIdxZ = SharedArray{Int}(size(IdxZ))
        sIdxZ[:] = IdxZ

        new(sZ, sW, M, sIdxZ)
    end

end
#
function DataSample(data, t::Int; v::Int=1, weight::Symbol=:counts)

    idxno0 = findall(data.counts[:,v,t].>0.)

    M = length(idxno0)
    Z = Array{Int8}(undef, data.L, M)
    IdxZ = Array{Int64}(undef, data.L, M)

    Wt = Array{Float64}(undef,M)

    if weight==:counts
        Wt=data.counts[idxno0,v,t]
    else
        Wt=ones(M)
    end
    Wt=Wt./sum(Wt)

    q=data.A
    q2=q*q

    for i in 1:M
        for j in 1:data.L
            Z[j,i] = data.variants[idxno0[i]].sequence[j]
            IdxZ[j,i] = (j-1) * q2 + q * (Z[j,i] - 1)
        end
    end

    DataSample(Z, Wt, M, IdxZ)
end
#
#one sample
function DataSample(Z::Array{Int,2}, W::Vector{Float64}; q::Int = Int(maximum(Z)))

    N,M = size(Z)
    @assert M == length(W)

    if sum(W) != 1
        W=W./sum(W)
    end

    IdxZ = Array{Int64}(undef, N, M)
    q2=q*q
    for i in 1:M
        for j in 1:N
            IdxZ[j,i] = (j-1) * q2 + q * (Z[j,i] - 1)
        end
    end

    DataSample(Z, W, M, IdxZ)
end

"""
    Workspace

Stores data and metaparameters.
"""
struct Workspace#{M}

    N::Int #length sequences part
    q::Int #number colors
    LL::Int #number of parameters


    S::Int  # number of variants
    T::Int  # number of samples

    #model::M # binding model
    #model0::M # zero round model
    #field_iter::I

    samples::Vector{DataSample} #data

    reg::NamedTuple #regularizzation

    opt_args::NamedTuple #optimization arguments

    """
        Workspace{U}(model, data, beta)
    """
    #function Workspace(model::M, data, bvec::Vector{Float64};
    function Workspace(N::Int, q::Int, LL::Int, S::Int, T::Int,
            samples::Vector{DataSample}, reg::NamedTuple, opt_args::NamedTuple)
            new(N, q, LL, S, T, samples, reg, opt_args)
    end

end

function Workspace(data;
    lambdaJE::Real=0.01,
    lambdaHE::Real=0.01,
    lambdaJG::Real=0.01,
    lambdaHG::Real=0.01,
    weight::Symbol=:counts,
    algorithm ::Symbol = :LD_LBFGS,
    verbose::Bool = true,
    epsconv::Float64 = 1e-10,
    maxit::Int = 10000)# where {M}

    @assert all(0 .≤ data.counts .< Inf)
    S, V, T = size(data.counts)
    @assert length(data.variants) == S
    N = data.L
    q = data.A
    LL = (N - 1)*q*q + q

    samples = DataSample[]
    for t = 1:T
        push!(samples,DataSample(data,t,v=V,weight=weight))
    end

    reg = (lambdaJE=lambdaJE, lambdaHE=lambdaHE, lambdaJG=lambdaJG, lambdaHG=lambdaHG)

    opt_args = (algorithm = algorithm, verbose = verbose,
                xatol = epsconv*1e2, xrtol = epsconv*1e2,
                fatol = epsconv, frtol = epsconv*1e-2, maxeval = maxit)

    Workspace(N, q, LL, S, T,
            samples, reg, opt_args)

end

function Workspace(s::DataSample;
    q::Int = 20,
    lambdaJE::Real=0.01,
    lambdaHE::Real=0.01,
    lambdaJG::Real=0.01,
    lambdaHG::Real=0.01,
    weight::Symbol=:counts,
    algorithm ::Symbol = :LD_LBFGS,
    verbose::Bool = true,
    epsconv::Float64 = 1e-10,
    maxit::Int = 10000
    )# where {M}


    S = s.M
    V = 1
    T = 1

    N = size(s.Z,1)
    LL = (N - 1) * q*q + q

    samples = [s]

    reg = (lambdaJE=lambdaJE, lambdaHE=lambdaHE, lambdaJG=lambdaJG, lambdaHG=lambdaHG)

    opt_args = (algorithm = algorithm, verbose = verbose,
                xatol = epsconv, xrtol = epsconv,
                fatol = epsconv, frtol = epsconv, maxeval = maxit)

    Workspace(N, q, LL, S, T,
            samples, reg, opt_args)

end


function Workspace(samples::Vector{DataSample};
    q::Int = 20,
    lambdaJE::Real=0.01,
    lambdaHE::Real=0.01,
    lambdaJG::Real=0.01,
    lambdaHG::Real=0.01,
    weight::Symbol=:counts,
    algorithm ::Symbol = :LD_LBFGS,
    verbose::Bool = true,
    epsconv::Float64 = 1e-10,
    maxit::Int = 10000
    )# where {M}

    S = samples[1].M
    V = 1
    T = length(samples)

    N = size(samples[1].Z,1)
    LL = (N - 1) * q*q + q

    reg = (lambdaJE=lambdaJE, lambdaHE=lambdaHE, lambdaJG=lambdaJG, lambdaHG=lambdaHG)

    opt_args = (algorithm = algorithm, verbose = verbose,
                xatol = epsconv, xrtol = epsconv,
                fatol = epsconv, frtol = epsconv, maxeval = maxit)

    Workspace(N, q, LL, S, T,
            samples, reg, opt_args)

end
