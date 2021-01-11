# JSOSolverTemplate

[![CircleCI](https://img.shields.io/circleci/project/abelsiqueira/JSOSolverTemplate.jl.svg?style=flat-square)](https://circleci.com/gh/abelsiqueira/JSOSolverTemplate.jl)

This template can be used to create a JSO-compliant solver.

## Framework

Create **two** repositories, one for the solver, one for the numerical experiments.
You can use this as a template for a JSO-compliant solver and [JSOExperimentsTemplate.jl](https://github.com/abelsiqueira/JSOExperimentsTemplate.jl) as template for the experiments.

## dev ../YourPackage.jl

Since you want to develop both packages at the same time, you should:

- open Julia in the experiments folder
- If not on VSCode, I recommend installing Revise and `julia> using Revise`
- `pkg> activate .` and `pkg> instantiate` if necessary
- `pkg> dev ../YourPackage.jl` where YourPackage.jl is this package.

Therefore you can edit your package and run the scripts for testing there, without leaving garbage on your package.

## Dr. Watson

You can use DrWatson.jl (link pending) to help in the numerical experiments part.
I don't use it here yet, because it is a little bit overkill for our usual tests.