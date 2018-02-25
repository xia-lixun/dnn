module Visual



mutable struct ProgressBar

    pph::Int64
    scaling::UnitRange{Int64}

    function ProgressBar(portions)
        a::Int128 = 0
        b::UnitRange{Int64} = 1:portions
        new(a, b)
    end
end


function rewind(t::ProgressBar)
    t.pph = 0
    nothing
end

function update(t::ProgressBar, i, n)
    
    pp = sum(i .>= n * (t.scaling/t.scaling[end]))
    delta = pp - t.pph
    delta > 0 && print(repeat(".", delta))
    t.pph = pp
    nothing
end




end  # module