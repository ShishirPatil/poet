# POET ILP Server Setup

This file describes how to setup the POET ILP server, either for use locally or as an external cloud service.

## Setting Up Gurobi (Optional)

The ILP solver defaults to using the COIN-OR CBC solver when Gurobi isn't available. However, since Gurobi is much faster, it is recommended to install it where possible.

### Acquiring a Free Academic Gurobi Web License

1. Create a free Gurobi account [here](https://pages.gurobi.com/registration). Make sure to specify the Academic user option.
2. Complete the rest of the Gurobi account creation process, which will include creating a password and verifying your email address.
3. Login to the Gurobi [Web License Manager](https://license.gurobi.com/) using your new account.
4. Create and download a new Web License file. This will be a `gurobi.lic` file that you will need in later steps, so keep note of where you save it.

## Option 1: Running the Server Locally

1. If using Gurobi, move the `gurobi.lic` file you downloaded in the previous step to your home directory (i.e. to `~/gurobi.lic`).
2. Clone this repository by running `git clone https://github.com/ShishirPatil/poet`.
3. Run `pip3 install -e .` in this repository's root directory to install the `poet-ai` package.
4. Run `cd poet-server` to navigate to the `poet-server` directory.
5. Run `pip3 install -r requirements.txt` to install the ILP server dependencies.
6. Finally, run `python3 server.py` to start the server.
    - You can optionally run `DEV=1 python3 server.py` to enable reload mode, which will automatically restart the server when you make changes to the code.
7. You can now make requests to the server at `http://localhost/solve`.

## Option 2: Running the Server Locally in a Docker Container

This folder also contains a Docker image that can be used to run the server. To run the server locally in a Docker container, follow these steps:

1. Ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed.
2. Clone this repository by running `git clone https://github.com/ShishirPatil/poet`.
3. If using Gurobi, move the `gurobi.lic` file you downloaded in the previous step to the `poet-server` directory of this repository (i.e. to `poet-server/gurobi.lic`).
4. Run `cd poet-server` to navigate to the `poet-server` directory.
5. Run `docker compose up --build` to build and start the Docker container.
6. You can now make requests to the server at `http://localhost/solve`.

## Option 3: Deploying the Dockerized Server on AWS

The Docker image described above is hosted on Docker Hub for convenient use on AWS. To run the server on AWS, follow these steps:

### 2. Creating an AWS EC2 Instance

1. Create an AWS account [here](https://aws.amazon.com/) if you don't already have one.
2. Open the AWS dashboard and navigate to the EC2 service. Use the instance creation wizard to create a new EC2 instance.
    - Make sure to use the default Amazon Linux 2 AMI
    - Set up SSH key pair authentication if needed. Move the `.pem` file to `~/.ssh/` and run `chmod 400 ~/.ssh/<keypair>.pem` to ensure that you can SSH into the instance.
    - You can use the default security group, but you may want to add a rule to allow SSH access from your IP address only.
3. Set up your SSH configuration file to automatically use the key pair. Add the following to `~/.ssh/config`:
    ```
    Host poet-server
        HostName <insert Public DNS URL from AWS>
        User ec2-user
        IdentityFile <insert path to your key pair>
    ```
4. If using Gurobi: move the `gurobi.lic` file you downloaded earlier to the EC2 instance by running the following command on your local computer: `scp path/to/local/gurobi.lic gurobi:~/`

### 3. Running the Gurobi Server with Docker

1. SSH into the EC2 instance by running `ssh poet-server`.
2. Run the following commands to install and start Docker:
    ```
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
    ```
3. Exit the SSH session and reconnect to ensure that the Docker group changes take effect.
4. Run the following commands to pull the image and start the Gurobi server:
    ```
    docker pull anishshanbhag/poet-server
    docker run -p 80:80 -v ~/gurobi.lic:/opt/gurobi/gurobi.lic anishshanbhag/poet-server
    ```
5. You should now be able to query the Gurobi server at the Public DNS URL from AWS.
