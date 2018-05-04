<p align="center">
<img src="assets/musicGenereClassification.png?raw=true" alt="MusicGenreClassification" width="250">
</p>


# MusicGenreClassification

Academic research in the field of **Deep Learning (Deep Neural Networks) and Sound Processing**, Tel Aviv University.

Featured in [Medium](https://medium.com/@matanlachmish/music-genre-classification-470aaac9833d).

## Abstract

This paper discuss the task of classifying the music genre of a sound sample.

## Introduction

When I decided to work on the field of sound processing I thought that genre classification is a parallel problem to the image classification. To my surprise I did not found too many works in deep learning that tackled this exact problem. One paper that did tackle this classification problem is Tao Feng’s paper [1] from the university of Illinois. I did learned a lot from this paper, but honestly, they results the paper presented were not impressive.

So I had to look on other, related but not exact papers. A very influential paper was Deep content-based music recommendation [2] This paper is about content-base music recommendation using deep learning techniques. The way they got the dataset, and the preprocessing they had done to the sound had really enlightened my implementation. Also, this paper was mentioned lately on “Spotify” blog [3]. Spotify recruited a deep learning intern that based on the above work implemented a music recommendation engine. His simple yet very efficient network made me think that Tao’s RBM was not the best approach and there for my implementation included a CNN instead like in the Spotify blog. One very important note is that Tao’s work published result only for 2,3 and 4 classes classification. Obviously he got really good result for 2 classes classification, but the more classes he tried to classify the poorer the result he got. My work classify the whole 10 classes challenge, a much more difficult task. A sub task for this project was to learn a new SDK for deep learning, I have been waiting for an opportunity to learn Google’s new TensorFlow[4]. This project is implemented in Python and the Machine Learning part is using TensorFlow.

## The Dataset

Getting the dataset might be the most time consuming part of this work. Working with music is a big pain, every file is usually a couple of MBs, there are variety of qualities and parameters of recording (Number of frequencies, Bits per second, etc…). But the biggest pain is copyrighting, there are no legit famous songs dataset as they would cost money. Tao’s paper based on a dataset called GTZAN[5]. This dataset is quit small (100 songs per genre X 10 genres = overall 1,000 songs), and the copyright permission is questionable. This is from my perspective one of the reasons that held him from getting better results. So, I looked up for generating more data to learn from. Eventually I found MSD[6] dataset (Million Song Dataset). It is a freely-available collection of audio features and metadata for a million contemporary popular music tracks. Around 280 GB of pure metadata. There is a project on top of MSD called tagtraum[7] which classify MSD songs into genres. The problem now was to get the sound itself, here is where I got a little creative. I found that one of the tags every song have in the dataset is an id from a provider called 7Digital[8]. 7Digital is a SaaS provider for music application, it basically let you stream music for money. I signed up to 7Digital as a developer and after their approval i could access their API. Still any song stream costs money, But I found out that they are enabling to preview random 30 seconds of a song to the user before paying for them. This is more than enough for my deep learning task, So I wrote “previewDownloader.py” that downloads for every song in the MSD dataset a 30 sec preview. Unfortunately I had only my laptop for this mission, so I had to settle with only 1% of the dataset (around 2.8GB).


The genres I am classifying are:
1. blues<br>
2. classical<br>
3. country<br>
4. disco <br>
5. hiphop<br>
6. jazz<br>
7. metal<br>
8. pop<br>
9. reggae<br>
10.rock<br>

<p align="center">
<img src="assets/music_popularity.png?raw=true" alt="Music genre popularity" width="500">
</p>

## Preprocessing the data

Having a big data set isn't enough, in oppose to image tasks I cannot work straight on the raw sound sample, a quick calculation: 30 seconds × 22050 sample/sec- ond = 661500 length of vector, which would be heavy load for a convention machine learning method.

Following all the papers I read and researching a little on acoustic analysis, It is quit obvious that the industry is using Mel-frequency cepstral coefficients (MFCC) as the feature vector for the sound sample, I used librosa[9] implementation.

MFCCs are derived as follows:
1. Take the Fourier transform of (a windowed excerpt of) a signal.
2. Map the powers of the spectrum obtained above onto the mel scale,
   using triangular overlapping windows.
3. Take the logs of the powers at each of the mel frequencies.
4. Take the discrete cosine transform of the list of mel log powers, as if it
   were a signal.
5. The MFCCs are the amplitudes of the resulting spectrum.

I had tried several window size and stride values, the best result I got was for size of 100ms and a stride of 40ms.

One more point was that Tao’s paper used MFCC features (step 5) while Sander used strait mel-frequencies (step 2).

<p align="center">
<img src="assets/mel_power_over_time.png?raw=true" alt="MEL ppower over time" width="650">
</p>

I tried both approaches and found out that I got extremely better results using just the mel-frequencies, but the trade-off was the training time of-course.
Before continue to building a network I wanted to visualise the preprocessed data set, I implemented this through the t-SNE[10] algorithm.Below you can see the t-SNE graph for MFCC (step 5) and Mel-Frequencies (step 2):
 
<p align="center">
<img src="assets/tsne_mfcc.png?raw=true" alt="t-SNE MFCC samples as genres" width="500">
</p>

<p align="center">
<img src="assets/tsne_mel_spec.png?raw=true" alt="t-SNE mel-spectogram samples as genres" width="500">
</p>
 
## The Graph
 
 After seeing the results Tao and Sander reached I decided to go with a convolu- tional neural network implementation. The network receive a 599 vector of mea-frequen- cy beans, each containing 128 frequencies which describe their window. The network consist with 3 hidden layers and between them I am doing a max pooling. Finally a fully connected layer and than softmax to end up with a 10 dimensional vector for our ten genre classes
 
<p align="center">
<img src="assets/nural_network.png?raw=true" alt="Nural Network" width="500">
</p>
 
 I did implement another network for MFCC feature instead of mel-frequencies, the only differences are in the sizes (13 frequencies per window instead of 128).
 
 Visualisation of various filters (from Sander’s paper):

<p align="center">
<img src="assets/filters.png?raw=true" alt="Filters visualization" width="250">
</p>

• Filter 14 seems to pick up vibrato singing.
• Filter 242 picks up some kind of ringing ambience.
• Filter 250 picks up vocal thirds, i.e. multiple singers singing
  the same thing, but the notes are a major third (4 semitones) apart.
• Filter 253 picks up various types of bass drum sounds.

## Results

As I explained in the introduction, the papers I based my work on did not solve the exact problem I did, for example Tao’s paper published results for classifying 2,3 and 4 classes (Genres). 
<p align="center">
<img src="assets/results_feng.png?raw=true" alt="Tao Feng's results" width="750">
</p>

I did looked for benchmarks outside the deep learning field and I found a paper titled “A BENCHMARK DATASET FOR AUDIO CLASSIFICATION AND CLUSTERING” [11]. This paper benchmark a very similar task to mine, the genres it classifies: Blues, Electronic, Jazz, Pop, HipHop, Rock, Folk, Alternative, Funk.

<p align="center">
<img src="assets/results_benchmark.png?raw=true" alt="Benchmark results" width="750">
</p>

### My results:
<p align="center">
<img src="assets/results_mine.png?raw=true" alt="My results" width="750">
</p>

## Code

### Documentation

• previewDownloader.py: 
USAGE: python previewDownloader.py [path to MSD data] 
This script iterate over all ‘.h5’ in a directory and download a 30 seconds sample from 7digital.

• preproccess.py: 
USAGE: python preproccess.py [path to MSD mp3 data] 
This script pre-processing the sound files. Calculating MFCC for a sliding window and saving the result in a ‘.pp’ file.

• formatInput.py: 
USAGE: python formatInput.py [path to MSD pp data] 
The script iterates over all ‘.pp’ files and generates ‘data’ and ‘labels’ that will be used as an input to the NN. 
Moreover, the script output a t-SNE graph at the end.

• train.py: 
USAGE: python train.py 
This script builds the neural network and feeds it with ‘data’ and ‘labels’.  When it is done it will save ‘model.final’.

### Complete Installation

<ul>
<li>Download the dataset files from https://www.dropbox.com/s/8ohx6m23co1qaz3/DataSet.zip?dl=0.</li>
<li>Unzip file</li>
<li>Place dataset files in the structure they are ordered in</li>
</ul>


## References

[1] Tao Feng, Deep learning for music genre classification, University of Illinois. https://courses.engr.illinois.edu/ece544na/fa2014/Tao_Feng.pdf
[2]Aar̈onvandenOord,SanderDieleman,BenjaminSchrauwen,Deepcontent- based music recommendation. http://papers.nips.cc/paper/5004-deep-content-based- music-recommendation.pdf
[3] SANDER DIELEMAN, RECOMMENDING MUSIC ON SPOTIFY WITH DEEP LEARNING, AUGUST 05, 2014. http://benanne.github.io/2014/08/05/spotify-cnns.html
[4] https://www.tensorflow.org
[5] GTZAN Genre Collection. http://marsyasweb.appspot.com/download/ data_sets/
[6] Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011. http://
labrosa.ee.columbia.edu/millionsong/
[7] Hendrik Schreiber. Improving genre annotations for the million song dataset. In
Proceedings of the 16th International Conference on Music Information Retrieval (IS- MIR), pages 241-247, 2015.
http://www.tagtraum.com/msd_genre_datasets.html
[8] https://www.7digital.com
[9] https://github.com/bmcfee/librosa
[10] http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html [11] Helge Homburg, Ingo Mierswa, Bu l̈ent Mo l̈ler, Katharina Morik and Michael
Wurst, A BENCHMARK DATASET FOR AUDIO CLASSIFICATION AND CLUSTERING, University of Dortmund, AI Unit. http://sfb876.tu-dortmund.de/PublicPublicationFiles/ homburg_etal_2005a.pdf

## Author

Matan Lachmish <sub>a.k.a</sub> <b>The Big Fat Ninja</b> <img src="assets/TheBigFatNinja.png?raw=true" alt="The Big Fat Ninja" width="13"><br>
https://thebigfatninja.xyz

### attribution

Icon made by <a title="Freepik" href="http://www.freepik.com">Freepik</a> from <a title="Flaticon" href="http://www.flaticon.com">www.flaticon.com</a>

## License

MusicGenreClassification is available under the MIT license. See the LICENSE file for more info.
