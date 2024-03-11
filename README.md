# PlmDCAtime


Several inference methods of the fitness landscape trained on sequencing samples from multiple rounds of a screening experiment, such as Deep Mutational Scans. The model comprises three main steps: selection, amplification, and sequencing. During the selection phase, protein variants are tested for a molecular function such as binding a target.  PlmDCAtime method can be applied to several experimental cases where a population of protein variants undergoes various rounds of selection and sequencing, without relying on the computation of variants enrichment ratios, and thus can be used even in cases of disjoint sequence samples.

The computational approach is described in Sesta et al. PLoS Comp. Biol.(2024). The packages have different method versions that integrates also the approach described in Sesta et al. IJMS (2021) which includes mutational steps as in Directed Evolution experiments.
Please cite [Inference of annealed protein fitness landscapes with AnnealDCA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011812) and [AMaLa: Analysis of Directed Evolution Experiments via Annealed Mutational Approximated Landscape](https://www.mdpi.com/1422-0067/22/20/10908) if you use (even partially) this code.



## Installation

This code is written in Julia language. To add `PlmDCAtime` package on Julia REPL run
```
Import("Pkg");Pkg.add("git@github.com:uguzzoni/PlmDCAtime.git")
```

## Quick start Example

```julia

using JLD2, PlmDCAtime

data = load("data/data_exp1.jld2")["data_experiment1"]

model = Model(
        (BiophysViabilityModel.ZeroEnergy(), BiophysViabilityModel.Kelsic_model()),
        zeros(2,1),
        zeros(1),
        reshape([false, true], 2, 1),
        reshape([true, false], 2, 1)
    ) 

history = learn!(model, data; epochs = 1:100, batchsize=256)

```

## Contributions

[Luca Sesta](https://github.com/lucasesta),  [Guido Uguzzoni](https://github.com/uguzzoni)([GU](mailto:guido.uguzzoni@gmail.com)), [Jorge Fernandez de Cossio Diaz](https://github.com/cossio)

## Maintainers
[Guido Uguzzoni](https://github.com/uguzzoni)([GU](mailto:guido.uguzzoni@gmail.com)), [Luca Sesta](https://github.com/lucasesta)

## License
[MIT license](LICENSE)
