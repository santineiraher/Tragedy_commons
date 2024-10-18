using Plots

function coopind_consumptionrate(propcoop, cooprate, indrate)
    return cooprate*propcoop + (1-propcoop)*indrate
end

function growthrate(R, r)
    return r * R * (1-R)
end

let 
    cooprate = 0.05
    indrate = 0.15
    r = 0.46
    condata = [coopind_consumptionrate(p, cooprate, indrate) for p in 0.0:0.01:1.0]
    growthdata = [growthrate(R, r) for R in 0.0:0.01:1.0]
    plot(0.0:0.01:1.0,condata)
    plot!(0.0:0.01:1.0,growthdata)
end