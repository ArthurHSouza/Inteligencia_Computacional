## Trabalho 1: A Genetic Algorithm Framework for Optimizing Three-Impulse Orbital Transfers with Poliastro Simulation
[experimentos_printando_hman.py](https://github.com/ArthurHSouza/Inteligencia_Computacional/blob/main/Trabalho1/AG/experimentos_printando_hman.py ): Arquivo que realiza uma iteração do AG e exibe os resultados em um gráfico, comparando com a solução analítica. Seções comentadas referentes ao cenário estudado ser geoestacionário.

[experimentos_repetidos.py](https://github.com/ArthurHSouza/Inteligencia_Computacional/blob/main/Trabalho1/AG/experimentos_repetidos.py ): Arquivo que realiza 200 iterações do AG e registra a média dos 'Delta V's e 'rp's. Parâmetros do AG estão ajustáveis em variáveis antes da execução e foram alterados manualmente para gerar a tabela da apresentação.

[experimentos_dias_singular.py](https://github.com/ArthurHSouza/Inteligencia_Computacional/blob/main/Trabalho1/AG/experimentos_dias_singular.py ): Semelhante a [experimentos_printando_hman.py](https://github.com/ArthurHSouza/Inteligencia_Computacional/blob/main/Trabalho1/experimentos_printando_hman.py ), porém com uma modificação feita para ponderar o custo de combustível em relação a duração em dias. Uma modificação anterior foi feita para incluir um limite fixo de dias e em seguida editada/removida para ser apenas o cálculo ponderado.

[experimentos_dias.py](https://github.com/ArthurHSouza/Inteligencia_Computacional/blob/main/Trabalho1/AG/experimentos_dias.py): Realiza 200 iterações do AG e registra as médias, porém com a ponderação em relação a dias e combustível. Também foi antes usado para testar um limite fixo de dias e em seguida modificado para usar o cálculo ponderado.

## Trabalho 2: Revisão com PSO Híbrido
[experimentos_PSOH.py](https://github.com/ArthurHSouza/Inteligencia_Computacional/blob/main/Trabalho1/PSO/experimentos_PSOH.py): Arquivo usado nos experimentos de PSO híbrido.
