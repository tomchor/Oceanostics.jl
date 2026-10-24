using Test
using YAML: YAML

# Two conventions that were kept by hand are checked here instead: that CI schedules every test group,
# and that fold markers are well formed. The group reads the repository's own files and runs no kernel,
# so it runs on GitHub Actions only.
const QA_REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

#+++ Test groups and the CI that schedules them
# Buildkite's documented build-matrix limits (Pipelines → Build matrix → "Matrix limits"): 25 elements
# per dimension and 50 jobs per `matrix` configuration. The GPU matrix has a single dimension, `group`,
# so its length is what reaches a limit first.
const BUILDKITE_MATRIX_ELEMENTS_PER_DIMENSION = 25
const BUILDKITE_MATRIX_JOBS = 50

# The groups the GPU pipeline leaves out on purpose; `.buildkite/gpu-pipeline.yml` says why.
const GROUPS_NOT_ON_GPU = Set(["perf_invariants", "quality_assurance"])

# Every group is an `if group == :name || group == :all` block, so `:all` is the one match that is not
# a group.
function declared_test_groups(root)
    runtests = read(joinpath(root, "test", "runtests.jl"), String)
    return setdiff(Set(String(m[1]) for m in eachmatch(r"group == :(\w+)", runtests)), ["all"])
end

function actions_test_groups(root)
    workflow = YAML.load_file(joinpath(root, ".github", "workflows", "ci.yml"))
    groups = String[]
    for job in values(workflow["jobs"]), step in job["steps"]
        env = get(step, "env", nothing)
        env isa AbstractDict && haskey(env, "TEST_GROUP") && push!(groups, env["TEST_GROUP"])
    end
    return groups
end

function buildkite_matrices(root)
    pipeline = YAML.load_file(joinpath(root, ".buildkite", "gpu-pipeline.yml"))
    return [step["matrix"]["setup"] for step in pipeline["steps"] if step isa AbstractDict && haskey(step, "matrix")]
end
#---

#+++ Fold markers
# A section opens with `#+++ <title>` and closes with `#---`, with exactly three `+` or `-` directly
# after the `#`, and every section is closed, nested ones included. A comment that starts with two or
# more `+` or `-` right after the `#`, or with `+` after `# `, is taken to be a marker; `# ---` is left
# alone, since in a Literate script it is a Markdown rule.
const FOLD_MARKER_ATTEMPT = r"^\s*#(\+{2,}|-{2,}|\s+\+{2,})"
const FOLD_OPEN = r"^\s*#\+\+\+ \S"
const FOLD_CLOSE = r"^\s*#---$"

function fold_marker_problems(file)
    problems = String[]
    open_sections = Int[] # line numbers of the sections still open
    for (n, line) in enumerate(eachline(file))
        occursin(FOLD_MARKER_ATTEMPT, line) || continue
        if occursin(FOLD_OPEN, line)
            push!(open_sections, n)
        elseif occursin(FOLD_CLOSE, line)
            isempty(open_sections) ? push!(problems, "line $n: `#---` closes no open section") : pop!(open_sections)
        else
            push!(problems, "line $n: malformed marker $(repr(strip(line)))")
        end
    end
    append!(problems, ["line $n: section is never closed" for n in open_sections])
    return problems
end

function julia_source_files(root)
    files = String[]
    for directory in ("src", "test", "docs"), (dir, _, names) in walkdir(joinpath(root, directory))
        # Documenter's output (`docs/build*`) and Literate's (`docs/src/generated`) are not sources
        any(part -> startswith(part, "build") || part == "generated", splitpath(relpath(dir, root))) && continue
        append!(files, joinpath(dir, name) for name in names if endswith(name, ".jl"))
    end
    return files
end
#---

@testset "Quality assurance" begin
    @info "  Testing that CI schedules every test group"
    @testset "CI schedules every test group" begin
        declared = declared_test_groups(QA_REPOSITORY_ROOT)
        @test !isempty(declared)

        on_actions = actions_test_groups(QA_REPOSITORY_ROOT)
        @test allunique(on_actions)

        not_on_actions = setdiff(declared, on_actions)
        isempty(not_on_actions) || @error "Test groups in test/runtests.jl with no job in .github/workflows/ci.yml" not_on_actions
        @test isempty(not_on_actions)

        stale_on_actions = setdiff(Set(on_actions), declared)
        isempty(stale_on_actions) || @error "Jobs in .github/workflows/ci.yml for groups test/runtests.jl does not define" stale_on_actions
        @test isempty(stale_on_actions)

        matrices = buildkite_matrices(QA_REPOSITORY_ROOT)
        @test !isempty(matrices)

        on_gpu = String[]
        for setup in matrices
            append!(on_gpu, setup["group"])
            for (dimension, elements) in setup
                n = length(elements)
                n <= BUILDKITE_MATRIX_ELEMENTS_PER_DIMENSION || @error "Buildkite matrix dimension exceeds the element limit" dimension n BUILDKITE_MATRIX_ELEMENTS_PER_DIMENSION
                @test n <= BUILDKITE_MATRIX_ELEMENTS_PER_DIMENSION
            end
            jobs = prod(length, values(setup))
            jobs <= BUILDKITE_MATRIX_JOBS || @error "Buildkite matrix exceeds the job limit" jobs BUILDKITE_MATRIX_JOBS
            @test jobs <= BUILDKITE_MATRIX_JOBS
        end
        @test allunique(on_gpu)

        expected_on_gpu = setdiff(declared, GROUPS_NOT_ON_GPU)
        not_on_gpu = setdiff(expected_on_gpu, on_gpu)
        isempty(not_on_gpu) || @error "Test groups in test/runtests.jl missing from the .buildkite/gpu-pipeline.yml matrix" not_on_gpu
        @test isempty(not_on_gpu)

        stale_on_gpu = setdiff(Set(on_gpu), expected_on_gpu)
        isempty(stale_on_gpu) || @error ".buildkite/gpu-pipeline.yml matrix entries that test/runtests.jl does not define, or that are kept off the GPU" stale_on_gpu
        @test isempty(stale_on_gpu)
    end

    @info "  Testing that fold markers are well formed and balanced"
    @testset "Fold markers are well formed and balanced" begin
        files = julia_source_files(QA_REPOSITORY_ROOT)
        @test !isempty(files)

        for file in files
            problems = fold_marker_problems(file)
            isempty(problems) || @error "Fold markers in $(relpath(file, QA_REPOSITORY_ROOT))" problems
            @test isempty(problems)
        end
    end
end
