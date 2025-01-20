include("scraping_tesouro.jl")
using Optim
using BusinessDays
using Plots
using Plots: text, default, scatter, scatter!, savefig, annotate!, plot!, plot
using Dates
using Statistics: mean
using InterestRates

# Tipos e constantes
struct BRLBond
    type::String
    maturity::String
    pu::Float64
    has_coupon::Bool
end

const BR_CAL = BusinessDays.BRSettlement()
const DAYS_IN_YEAR = 252.0
const FACE_VALUE = 1000.0
const COUPON_ANNUAL_RATE = 0.10  # Taxa anual padrão para títulos com cupom

# Configuração do tema do gráfico
ENV["GKSwstype"] = "100"
default(fontfamily="Computer Modern", legend=:topright, grid=true)

# Funções auxiliares
has_coupon(bond_type::String) = occursin("Juros Semestrais", bond_type)
to_date(date_str::String) = Date(date_str, "d/m/Y")
format_date(date::Date) = Dates.format(date, "d/m/Y")
yearfrac(start_date::String, end_date::String) = 
    BusinessDays.bdayscount(BR_CAL, to_date(start_date), to_date(end_date)) / DAYS_IN_YEAR

# Gera datas de cupom
function generate_coupon_dates(start_date::String, maturity_date::String)
    maturity = to_date(maturity_date)
    start = to_date(start_date)
    
    @assert Dates.day(maturity) ∈ [1, 15] "Data de vencimento deve ser dia 1 ou 15"
    
    coupon_dates = maturity:-Month(6):start |> collect |> reverse
    format_date.(coupon_dates)
end

# Calcula fluxo de caixa
function calculate_cash_flow(face_value::Float64, annual_rate::Float64, 
                           start_date::String, maturity_date::String, is_bullet::Bool=false)
    is_bullet && return [(maturity_date, face_value)]
    
    semiannual_rate = (1 + annual_rate)^(1/2) - 1
    coupon_dates = generate_coupon_dates(start_date, maturity_date)
    coupon_value = face_value * semiannual_rate
    
    [(date, i == length(coupon_dates) ? coupon_value + face_value : coupon_value) 
     for (i, date) in enumerate(coupon_dates)]
end

# Calcula taxa implícita
function calculate_implied_rate(pu::Float64, pu_date::String, cash_flow::Vector)
    pu_dt = to_date(pu_date)
    
    objective(rate) = begin
        daily_rate = (1 + rate)^(1/DAYS_IN_YEAR) - 1
        sum(cash_flow) do (date, value)
            days = BusinessDays.bdayscount(BR_CAL, pu_dt, to_date(date))
            value / (1 + daily_rate)^days
        end |> x -> abs(x - pu)
    end
    
    result = optimize(objective, 0.0, 1.0, Brent())
    !Optim.converged(result) && error("Não foi possível convergir para taxa implícita")
    
    Optim.minimizer(result)
end

# Constrói curva zero-coupon
function build_zero_coupon_curve(zero_coupon_bonds::Vector{BRLBond}, ref_date::String)
    zero_coupon_curve = Dict{Date, Float64}()
    for bond in zero_coupon_bonds
        yield = calculate_implied_rate(bond.pu, ref_date, [(bond.maturity, FACE_VALUE)])
        zero_coupon_curve[to_date(bond.maturity)] = yield
    end
    return zero_coupon_curve
end

# Ajusta curva iterativamente usando títulos com cupom
function iterative_curve_adjustment(
    zero_coupon_curve::Dict{Date, Float64},
    coupon_bonds::Vector{BRLBond},
    ref_date::String
)
    adjusted_curve = deepcopy(zero_coupon_curve)
    
    for bond in coupon_bonds
        cash_flow = calculate_cash_flow(FACE_VALUE, COUPON_ANNUAL_RATE, ref_date, bond.maturity, false)
        observed_price = bond.pu
        
        function objective(rate)
            adjusted_curve[to_date(bond.maturity)] = rate
            price = price_with_coupon(cash_flow, adjusted_curve, ref_date)
            return abs(price - observed_price)
        end
        
        result = optimize(objective, 0.0, 1.0, Brent())
        new_rate = Optim.minimizer(result)
        adjusted_curve[to_date(bond.maturity)] = new_rate
    end
    
    return adjusted_curve
end

# Constrói curva completa usando todos os títulos
function build_complete_yield_curve(bonds::Vector{BRLBond}, ref_date::String)
    zero_coupon_bonds = filter(b -> !b.has_coupon, bonds)
    coupon_bonds = filter(b -> b.has_coupon, bonds)
    
    zero_coupon_curve = build_zero_coupon_curve(zero_coupon_bonds, ref_date)
    complete_curve = iterative_curve_adjustment(zero_coupon_curve, coupon_bonds, ref_date)
    
    ref_dt = to_date(ref_date)
    dates = collect(keys(complete_curve)) |> sort
    days = [BusinessDays.bdayscount(BR_CAL, ref_dt, d) for d in dates]
    rates = [complete_curve[d] for d in dates]
    
    curve = InterestRates.IRCurve("curva-completa",
        InterestRates.BDays252(:Brazil),
        InterestRates.ExponentialCompounding(),
        InterestRates.FlatForward(),
        ref_dt,
        days,
        rates)
    
    daily_curve = Dict{Date, Float64}()
    current_date = first(dates)
    last_date = last(dates)
    
    while current_date <= last_date
        if BusinessDays.isbday(BR_CAL, current_date)
            rate = InterestRates.zero_rate(curve, current_date)
            daily_curve[current_date] = rate
        end
        current_date = current_date + Day(1)
    end
    
    return daily_curve
end

# Cria curva interpolada
function create_curve(zero_coupon_curve::Dict{Date, Float64}, ref_date::Date)
    dates = collect(keys(zero_coupon_curve)) |> sort
    rates = [zero_coupon_curve[d] for d in dates]
    days = [BusinessDays.bdayscount(BR_CAL, ref_date, d) for d in dates]
    
    InterestRates.IRCurve("curva-zero",
        InterestRates.BDays252(:Brazil),
        InterestRates.ExponentialCompounding(),
        InterestRates.FlatForward(),
        ref_date,
        days,
        rates)
end

# Precifica título com cupom usando curva interpolada
function price_with_coupon(cash_flow::Vector{Tuple{String, Float64}}, zero_coupon_curve::Dict{Date, Float64}, ref_date::String)
    ref_dt = to_date(ref_date)
    curve = create_curve(zero_coupon_curve, ref_dt)
    
    price = 0.0
    for (date_str, value) in cash_flow
        date = to_date(date_str)
        df = InterestRates.discountfactor(curve, date)
        price += value * df
    end
    return price
end

# Plota curva completa
function plot_complete_curve(complete_curve::Dict{Date, Float64}, ref_date::String; output_file::Union{String,Nothing}=nothing)
    dates = collect(keys(complete_curve)) |> sort
    rates = [complete_curve[d] * 100 for d in dates]
    days = [BusinessDays.bdayscount(BR_CAL, to_date(ref_date), d) / DAYS_IN_YEAR for d in dates]
    
    min_rate = minimum(rates)
    max_rate = maximum(rates)
    min_days = minimum(days)
    max_year = ceil(maximum(days))  # Definir max_year antes de usar no plot
    
    p = plot(days, rates,
        xlabel="Anos",
        ylabel="Taxa (%)",
        title="Curva de Juros Pré - Tesouro Direto ($ref_date) - Manhã",
        linewidth=2,
        color=:blue,
        size=(1600, 600),  # Aumentado para 1600px de largura
        margin=20Plots.mm,
        ylims=(min_rate * 0.8, max_rate * 1.2),
        xlims=(min_days, max_year),
        legend=false
    )
    
    # Configuração do grid a partir do primeiro ponto da curva
    start_year = ceil(min_days * 2) / 2  # Arredonda para o próximo múltiplo de 0.5
    xticks_values = start_year:0.5:max_year
    plot!(p, xticks=xticks_values, grid=:x, gridstyle=:dot, gridalpha=0.3)
    
    # Annotate rates at each grid point com verificação de datas existentes
    ref_dt = to_date(ref_date)
    curve_dates = sort(collect(keys(complete_curve)))
    
    for x in xticks_values
        days_count = round(Int, x * DAYS_IN_YEAR)
        target_date = BusinessDays.advancebdays(BR_CAL, ref_dt, days_count)
        
        # Encontra a data mais próxima usando busca binária
        idx = searchsortedfirst(curve_dates, target_date)
        if idx > length(curve_dates)
            idx = length(curve_dates)
        elseif idx > 1
            # Verifica qual data está mais próxima
            if (target_date - curve_dates[idx-1]) < (curve_dates[idx] - target_date)
                idx -= 1
            end
        end
        
        if 1 <= idx <= length(curve_dates)
            rate = complete_curve[curve_dates[idx]] * 100
            annotate!(p, x, rate, text("$(round(rate, digits=2))%", 8, :bottom, :black))
        end
    end
    
    output_file !== nothing && savefig(p, output_file)
    p
end

# Gera e plota curva
ref_date = "17/01/2025"
df = load_treasury_data()
filtered_df = filter_treasury_bonds(df, ref_date, type="PRE")
bonds = [BRLBond(
    row["Tipo Titulo"],
    row["Data Vencimento"],
    mean([parse(Float64, replace(string(row[k]), "," => ".")) 
          for k in ["PU Compra Manha", "PU Venda Manha"]]),
    has_coupon(row["Tipo Titulo"])
) for row in eachrow(filtered_df)]

complete_curve = build_complete_yield_curve(bonds, ref_date)
plot_complete_curve(complete_curve, ref_date, output_file="yield_curve.png")
