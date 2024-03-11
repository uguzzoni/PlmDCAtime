module PlmDCA
using GaussDCA,SharedArrays,Distributed,Printf, LinearAlgebra, Statistics

using NLopt

export PlmOut, plmdca, plmdcasym, mutualinfo

using JLD2

include("types.jl")
include("utils.jl")
include("plmdca_asym.jl")
include("emdca_sym.jl")
#include("decimation_asym.jl")
include("plmdca_sym.jl")
include("decimation_sym.jl")
include("mi.jl")
end
