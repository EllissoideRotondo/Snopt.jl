using Snopt
using Test

if !Snopt.has_snopt()
    @info "Snopt.jl: SNOPT library not found, skipping all tests."
    exit(0)
end

if Sys.iswindows()
    #@info "Snopt.jl: Windows is not yet supported, skipping all tests."
    #exit(0)
end

@info "Running tests with $(Snopt.libsnopt7)"

@testset "Test examples" begin
    include("snopt_tests.jl")
end
