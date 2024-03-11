mutable struct PlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    xtol::Float64
    maxit::Int
    function PlmAlg(method,verbose, epsconv,xtol, maxit)
        new(method, verbose, epsconv, xtol, maxit)
    end
    function PlmAlg(method,verbose, epsconv, maxit)
        new(method, verbose, epsconv, 1e-15, maxit)
    end
    function PlmAlg(method)
        new(method, true, 1e-15, 1e-15,  10000)
    end
    # boolmask::Union{SharedArray{Bool,2},Nothing}
    # function PlmAlg(method,verbose, epsconv, maxit, boolmask)
    #     if boolmask != nothing
    #         sboolmask = SharedArray{Bool}(size(boolmask))
    #         sboolmask[:] = boolmask
    #         new(method, verbose, epsconv, maxit, sboolmask)
    #     else
    #         new(method, verbose, epsconv, maxit, nothing)
    #     end
    # end
end
#
# mutable struct PlmAlgBeta
#     method::Symbol
#     verbose::Bool
#     epsconvbeta::Float64
#     maxitbeta::Int
# end

struct PlmOut{N}
    pslike::Union{Vector{Float64},Float64}
    Jtensor::Array{Float64,N}
    htensor::Array{Float64,2}
    score::Array{Tuple{Int, Int, Float64},1}
end

struct PlmVar
    N::Int
    M::Int
    q::Int
    q2::Int
    #gaugecol::Int
    lambdaJE::Float64
    lambdaHE::Float64
    lambdaJG::Float64
    lambdaHG::Float64
    lambdaJM::Float64
    lambdaHM::Float64
    Z:: SharedArray{Int,2}
    IdxZ::SharedArray{Int,2} #partial index computation for speed up energy calculation
    W::SharedArray{Float64,1}

    function PlmVar(N,M,q,q2,lambdaJE,lambdaHE,lambdaJG,lambdaHG,lambdaJM,lambdaHM,Z,IdxZ,W)
        sZ = SharedArray{Int}(size(Z))
        sZ[:] = Z
        sW = SharedArray{Float64}(size(W))
        sW[:] = W
        sIdxZ = SharedArray{Int}(size(IdxZ))
        sIdxZ[:] = IdxZ
        new(N,M,q,q2,lambdaJE,lambdaHE,lambdaJG,lambdaHG,lambdaJM,lambdaHM,sZ,sIdxZ,sW)
    end
end

struct DecVar{N}
    fracdec::Float64
    fracmax::Float64
    blockdecimate::Bool
    dmask::SharedArray{Bool,N}
    function DecVar{N}(fracdec, fracmax, blockdecimate, dmask) where N
        sdmask = SharedArray{Bool}(size(dmask))
        sdmask[:] = dmask
        new(fracdec, fracmax, blockdecimate, sdmask)
    end
end
