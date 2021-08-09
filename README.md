# Various GAN Models

GANの様々なモデルを一元管理し、コマンドラインにより使い分けを可能にする

## Supported Models

### Modules

- MLP (fc_module)

### Activation Modules

- Softmax
- L2Softmax
- ArcFace
- CosFace

## Supported Data

- [宗教画の分類](https://prob.space/competitions/religious_art) のデータ

## Set Up

* download data
  
  最初に次を実行し、学習に必要なデータをダウンロード・配置する

    ```bash
    $ make
    ```

* docker build image & create container

  dockerのコンテナを立ち上げる

    ```bash
    $ docker-compose up -d --build
    ```

## Environment

## Command

### usage

学習方法あれこれ

### docker usage

以下はdockerの基本的な使用方法

* start

    ```bash
    $ docker-compose start
    ```

* exec

    ```bash
    $ docker-compose exec python3 /bin/bash
    ```

* stop

    ```bash
    $ docker-compose stop
    ```

* restart

    ```bash
    $ docker-compose restart
    ```

* remove container

    ```bash
    $ docker-compose down
    ```

* remove image

    ```bash
    $ docker-compose down --rmi all
    ```
