# indication dutilities
mutable struct indicator
    pph::Int64
    scaling::UnitRange{Int64}

    function indicator(portions)
        pph_v::Int128 = 0
        scale_v::UnitRange{Int64} = 1:portions
        new(pph_v, scale_v)
    end
end


function rewind(t::indicator)
    t.pph = 0
end

function update(t::indicator, i, n)
    pp = sum(i .>= n * (t.scaling/t.scaling[end]))
    delta = pp-t.pph
    delta > 0 && print(repeat(".", delta))
    t.pph = pp
end