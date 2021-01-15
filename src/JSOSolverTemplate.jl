module JSOSolverTemplate

# standard lib
using LinearAlgebra

# JSO packages
using NLPModels
using SolverTools

include("uncsolver.jl")
include("gradienteCon.jl")
include("STCG.jl")

end # module
