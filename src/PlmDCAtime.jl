module PlmDCAtime


using Distributed
using IterTools
using LinearAlgebra
using NLopt
using Printf
using Random
using SharedArrays
using Statistics

module bEG


import SharedArrays: SharedArray
import Distributed: @distributed

export learn_par, DataSample, Workspace

include("bEG/workspace_bEG.jl")
include("bEG/parameters_bEG.jl")
include("bEG/optimize_bEG.jl")
include("bEG/gradient_bEG.jl")
include("bEG/learn_bEG.jl")

end

using .bEG

module bEnuHD

import SharedArrays: SharedArray
import Distributed: @distributed

export learn_par, DataSample, Workspace

include("bEnuHD/workspace_bEnuHD.jl")
include("bEnuHD/parameters_bEnuHD.jl")
include("bEnuHD/optimize_bEnuHD.jl")
include("bEnuHD/gradient_bEnuHD.jl")
include("bEnuHD/learn_bEnuHD.jl")

end

using .bEnuHD

module bEnuMG

import SharedArrays: SharedArray
import Distributed: @distributed

export learn_par, DataSample, Workspace

include("bEnuMG/workspace_bEnuMG.jl")
include("bEnuMG/parameters_bEnuMG.jl")
include("bEnuMG/optimize_bEnuMG.jl")
include("bEnuMG/gradient_bEnuMG.jl")
include("bEnuMG/learn_bEnuMG.jl")

end

#using .bEnuMG

# module bEnuMG0

#     using GaussDCA,SharedArrays,Distributed,Printf,LinearAlgebra, Statistics,
#     IterTools,Random

#     include("bEnuMG0/workspace_bEnuMG0.jl")
#     include("bEnuMG0/parameters_bEnuMG0.jl")
#     include("bEnuMG0/optimize_bEnuMG0.jl")
#     include("bEnuMG0/gradient_bEnuMG0.jl")
#     include("bEnuMG0/learn_bEnuMG0.jl")

# end

# using .bEnuMG0

module bEvcG

import SharedArrays: SharedArray
import Distributed: @distributed

export learn_par, DataSample, Workspace

include("bEvcG/workspace_bEvcG.jl")
include("bEvcG/parameters_bEvcG.jl")
include("bEvcG/optimize_bEvcG.jl")
include("bEvcG/gradient_bEvcG.jl")
include("bEvcG/learn_bEvcG.jl")

end

using .bEvcG

end