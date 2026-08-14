using SNOPT
using Test

if !SNOPT.has_snopt()
    @testset "Missing SNOPT library" begin
        @test !SNOPT.has_snopt()
        @test SNOPT.libsnopt7 == ""
        withenv("SNOPTDIR" => tempname()) do
            @test SNOPT.find_snopt_lib() == ""
        end

        err = try
            SNOPT.initialize("", "")
        catch err
            err
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("SNOPT library not loaded", msg)
        @test occursin("SNOPTDIR", msg)
    end
    if get(ENV, "SNOPT_REQUIRE_LIBRARY", "") == "true"
        error("SNOPT_REQUIRE_LIBRARY=true but no loadable libsnopt7 was found; " *
              "solver tests cannot run.")
    end
    @warn "SNOPT.jl: SNOPT library not found — ALL SOLVER TESTS SKIPPED. " *
          "Set SNOPTDIR (or install libsnopt7 on the loader path) to run the full suite."
    exit(0)
end

@info "Running tests with $(SNOPT.libsnopt7)"

@testset "Test examples" begin
    include("snopt_tests.jl")
end
