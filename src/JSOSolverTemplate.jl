module JSOSolverTemplate

# standard lib
using LinearAlgebra

# JSO packages
using NLPModels
using SolverTools

include("uncsolver.jl")
include("gradienteCon.jl")
include("STCG.jl")
include("bfgs.jl")
include("newtonar.jl")
include("newtonmodar.jl")

end # module
