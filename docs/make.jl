using Documenter
using SNOPT

DocMeta.setdocmeta!(SNOPT, :DocTestSetup, :(using SNOPT); recursive = true)

makedocs(;
    modules = [SNOPT],
    authors = "Alex Pascarella",
    sitename = "SNOPT.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://EllissoideRotondo.github.io/SNOPT.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "High-level interface" => "interface.md",
        "Low-level interface" => "lowlevel.md",
        "Examples" => "examples.md",
        "Optimization.jl integration" => "optimization.md",
        "API reference" => "api.md",
    ],
    # Every exported symbol must appear in an @docs block.
    checkdocs = :exports,
    # The SNOPT shared library is unavailable in CI, so no solves run during the
    # build; doctests are disabled because their output depends on the solver.
    doctest = false,
)

deploydocs(;
    repo = "github.com/EllissoideRotondo/SNOPT.jl",
    devbranch = "main",
    push_preview = true,
)
