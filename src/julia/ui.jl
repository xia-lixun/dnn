module UI



mutable struct Progress
    pph::Int64
    scaling::UnitRange{Int64}

    function Progress(portions)
        pph_v::Int128 = 0
        scale_v::UnitRange{Int64} = 1:portions
        new(pph_v, scale_v)
    end
end


function rewind(t::Progress)
    t.pph = 0
    nothing
end

function update(t::Progress, i, n)
    
    pp = sum(i .>= n * (t.scaling/t.scaling[end]))
    delta = pp-t.pph
    delta > 0 && print(repeat(".", delta))
    t.pph = pp
    nothing
end



# module
end