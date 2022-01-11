# temperature_forcasting_LoRa
Deep Learning architecture for temperature forecasting in an IoT LoRa based system

This is the implementation of paper : ***"Deep Learning architecture for temperature forecasting in an IoT LoRa based system"***


This study adopts recurrent neural networks (RNN) with its Long Short-Term Memory (LSTM) architecture to predict the ambient temperature (TA). The prediction is based on meteorological data retrieved from IoT stations, these IoT stations consist of different components such as sensors to capture the temperature, humidity and some gases in the air, and send them to the basic station with LoRa protocol. We formulate the TA prediction problem as a time series regression problem. LSTM is a particular type of recurrent neural network, which has a strong ability to model the temporal relationship of time series data and can well manage the problem of long-term dependency. The proposed network architecture consists of two types of hidden layers: LSTM layer and full connected dense layer. The LSTM layer is used to model the time series relationship. The fully connected layer is used to map the output of the LSTM layer to a final prediction. To confirm the effectiveness of the proposed model, we perform tests on data collected by our own IoT system on Tangier. In addition, we show all the results in a web interface.

*==> For more technical details we shared the thesis report in addition to the published paper.*

**REFERNCES**
* https://dl.acm.org/doi/10.1145/3320326.3320375
* https://www.researchgate.net/publication/333258384_Deep_Learning_architecture_for_temperature_forecasting_in_an_IoT_LoRa_based_system

**CITATION** Ben Abdel Ouahab Ikram, Boudhir Anouar Abdelhakim, Astito Abdelali, Bassam Zafar, and Bouhorma Mohammed. 2019. Deep Learning architecture for temperature forecasting in an IoT LoRa based system. In Proceedings of the 2nd International Conference on Networking, Information Systems & Security (NISS19). Association for Computing Machinery, New York, NY, USA, Article 43, 1–6. DOI:https://doi.org/10.1145/3320326.3320375
