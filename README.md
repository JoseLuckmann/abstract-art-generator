# Abstract Art Generator 🧑🏻‍🎨

Bem-vindo ao repositório do Abstract Art Generator, uma aplicação de ciência de dados e aprendizado de máquina que utiliza redes GAN (Generative Adversarial Networks) e técnicas de upscaling para gerar arte abstrata única e impressionante! Confira a aplicação ao vivo em [abstract-art-generator.streamlit.app](https://abstract-art-generator.streamlit.app/).

## O que é o Abstract Art Generator? ⁉️

O Abstract Art Generator é um projeto que combina conceitos de Deep Learning e Processamento de Imagens para criar arte abstrata. Utilizamos dois modelos principais:

1. **Gerador (GAN)**: Um modelo baseado em redes generativas adversárias para gerar imagens abstratas iniciais.
2. **Upscaling**: Um modelo para melhorar a resolução das imagens geradas, tornando-as mais detalhadas e visualmente atraentes.

## Como Funciona? ⚙️

- **Gerador (GAN)**: O modelo GAN é treinado com um conjunto de imagens abstratas para aprender a gerar novas imagens. No nosso caso, utilizamos uma estrutura de rede convolucional transposta para gerar imagens a partir de um vetor de ruído.
- **Upscaling**: Após a geração da imagem, aplicamos um modelo de upscaling para aumentar a resolução da imagem. Esse modelo utiliza convoluções e operações de upsample para aumentar os detalhes da imagem.

## Tecnologias Utilizadas 💻

- **Python**: Linguagem principal para desenvolvimento de modelos e aplicação Streamlit.
- **PyTorch**: Biblioteca de Deep Learning para construção e treinamento dos modelos.
- **Streamlit**: Framework para criação rápida de aplicativos web para Data Science.
- Outras bibliotecas: `numpy`, `matplotlib`, etc.

## Estrutura do Projeto 📂

```plaintext
seu_projeto/
│
├── app.py                 # Aplicativo Streamlit
├── models/                # Modelos treinados (GAN e Upscaling)
├── notebooks/             # Notebooks de treino e teste
├── utils/                 # Códigos auxiliares
└── requirements.txt       # Dependências
```

## Executando Localmente 🏃🏻‍♂️

Para executar o projeto localmente:

1. Clone o repositório.
2. Instale as dependências usando `pip install -r requirements.txt`.
3. Execute `streamlit run app.py`.

## Contribuições 🤝

Contribuições são sempre bem-vindas! Se você tem uma ideia para melhorar a aplicação ou os modelos, sinta-se à vontade para criar um pull request ou abrir uma issue.

## Licença 📃

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

Desenvolvido com 💓 por José Henrique Luckmann
