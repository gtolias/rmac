# R-MAC and integral max pooling

This is a Matlab/MEX package that we provide to implement the methods
of our ICLR 2016 paper (paper homepage: http://arxiv.org/abs/1511.05879).

<img src="http://cmp.felk.cvut.cz/~toliageo/images/thumbs/aml2.png" height="250"/>

## What is it?
This code implements

1) MAC and R-MAC image representation
2) Image retrieval based on MAC and R-MAC
3) Localization re-ranking (AML) and query expansion (QE)
4) Evaluation on Oxford5k and Paris6k test datasets

## Citation

```
@conference{tolias2016rmac,
  title={Particular object retrieval with integral max-pooling of CNN activations},
  author={Tolias, Giorgos and Sicre, Ronan and J{\'e}gou, Herv{\'e}},
  journal={International Conference on Learning Representations},
  year={2016}
}
```

## Prerequisites

The prerequisites are:
- Images of Oxford5k and Paris6k datasets: http://www.robots.ox.ac.uk/~vgg/data/

- MatConvNet MATLAB toolbox 1.0-beta25 is automatically downloaded and compiled
  http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz

- Pre-trained CNN models. In our paper we have used AlexNet and VGG16. The mat files containing the models are downloaded in the code. The current files on matconvnet website are slightly different from the ones we used (older versions downloaded from matconvnet website).

- The code is tested with MATLAB 9.2 (R2017a) on Linux.

## Execution

1) Run the following script
```
>> test
```

## License

This is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
