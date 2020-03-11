# Similarity-Based-Chained-Transfer-Learning

Code for publication: Similarity-Based Chained Transfer Learning for Energy Forecasting With Big Data

Smart meter popularity has resulted in the ability to collect big energy data and has created opportunities for large-scale energy forecasting. 
Machine Learning (ML) techniques commonly used for forecasting, such as neural networks, involve computationally intensive training typically with data from a single building or a single aggregated load to predict future consumption for that same building or aggregated load. 
With hundreds of thousands of meters, it becomes impractical or even infeasible to individually train a model for each meter. 
Consequently, this paper proposes Similarity-Based Chained Transfer Learning (SBCTL), an approach for building neural network-based models for many meters by taking advantage of already trained models through transfer learning. 
The first model is trained in a traditional way whereas all other models transfer knowledge from the existing models in a chain-like manner according to similarities between energy consumption profiles. 
A Recurrent Neural Network (RNN) was used as the base forecasting model, two initialization techniques were considered, and different similarity measures were explored. 
The experiments show that SBCTL achieves accuracy comparable to traditional ML training while taking only a fraction of time.

If you have questions, feel free to email:
ytian285@gmail.com

[1] Y. Tian, L. Sehovac and K. Grolinger, "Similarity-Based Chained Transfer Learning for Energy Forecasting With Big Data," in IEEE Access, vol. 7, pp. 139895-139908, 2019.
doi: 10.1109/ACCESS.2019.2943752
keywords: {Big Data;energy consumption;learning (artificial intelligence);load forecasting;power engineering computing;recurrent neural nets;smart meters;computationally intensive training;aggregated load;transfer learning;neural network-based models;trained models;energy consumption profiles;recurrent neural network;similarity measures;ML training;Big Data;smart meter popularity;big energy data;energy forecasting;similarity-based chained transfer learning;SBCTL;Forecasting;Smart meters;Training;Meters;Buildings;Computational modeling;Recurrent neural networks;Big data;deep learning;energy forecasting;gated recurrent units;recurrent neural network;smart meters;transfer learning},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8848386&isnumber=8600701

