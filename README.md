# Trabalho da disciplina
Nesse diretório é possível encontrar os códigos utilizados para a realização da disciplina Tópicos em Aprendizado de Máquina (Deep Learning) ministrada no segundo semestre de 2025

# Resumo do trabalho
Com o aumento das leis de proteção de dados em todo o mundo, a necessidade de disponibilizar conjuntos de dados anonimizados para modelos tornou-se cada vez mais importante. Isso levanta a hipótese de que uma rede neural pode aprender as características do processo de anonimização em vez das características do objeto a ser identificado, reduzindo potencialmente sua eficácia em situações reais. Neste trabalho é analisado o desempenho de três modelos com diferentes profundidades – AlexNet, VGG e ResNet – na tarefa de classificação de pessoas sob diferentes técnicas de anonimização. Os experimentos foram conduzidos treinando as redes com o mesmo conjunto de imagens de pessoas do conjunto de dados Microsoft Common Objects in Context, com a única diferença sendo o tipo de anonimização aplicada: sem censura, borramento e o de caixa preta. Os resultados foram avaliados com base no recall de cada modelo no conjunto de teste composto por imagens não anonimizadas. Como resultado, as redes apresentaram maior dificuldade em classificar imagens contendo pessoas quando treinadas com dados anonimizados.

# Arquivos

## paper.tex
Este arquivo é a formatação LATEX do artigo de apresentação do trabalho da disciplina

## dataPrep
Nesse diretório encontram-se os códigos utilizados para a preparação do dataset
- Primeiramente foi feito o download das anotações a partir do 'tutorial' em: https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9
- **PersonWithHead.py**: Faz o download de todas as imagens que possuem as anotações de keypoints e cria um novo json com as informações dessas imagens
- **SubsetNotPerson.py**: Faz o download de todas as imagens anotadas com as classes informadas e cria um novo json com as informações dessas imagens
- **Crop.py**: Recorta as imagens de acordo com a bounding box indicada nas anotações do json e as salva em um novo diretório.
- **filtraOriginalByCrop.py**: Seleciona as imagens originais a partir das selecionadas pelo crop
- **Redistribui.py**: Redistribui o dataset inicial de treino em treino e teste
- **blur.py**: cria duas cópias das imagens originais, uma anonimizando as imagens com gaussian blur e outra com uma caixa preta na região do rosto a partir das anotações de keypoint  
> Para a utilização dos códigos altere os paths para as imagens e anotações quando necessário

## train
Nesse diretório encontram-se os códigos utilizados para o treinamento dos modelos utilizados no experimento proposto
- **Config.py**: Declara a classe de configuração com os parâmetro utilizados nos treinamentos como: path para os dados, número de épocas, tamanho de batch, quantidade de épocas de paciência e transformação dos dados
- **Alex.py**: Script para o experimento com o modelo AlexNet
- **Vgg.py**: Script para o experimento com o modelo VGG16
- **resNet.py**: Script para o experimento com o modelo resnet18

## logs
Nesse diretório estão os arquivos de logging resultante dos experimentos
