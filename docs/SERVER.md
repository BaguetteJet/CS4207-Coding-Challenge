# Rockhill Server

This server cluster contains:

- CPU: 2x AMD EPYC 9124
- GPU: 2x Nvidia H100 96 GB
- RAM: 503 GB

It is only accessible through lab computers on the **ul.campus** network.

## Connect

Log into lab computer using you student email and password.

SSH into the server using the assigned group login.

```
ssh group19@rockhill.csis.ul.ie
```

### Setup Project

Create project and data folder.

```
mkdir -p ~/CS4207-Coding-Challenge/data
```

Dataset files are stored on server under `/home/cs4207/data/`.

Create symbolic link shortcuts for the dataset files.

```
ln -s /home/cs4207/data/* ~/CS4207-Coding-Challenge/data
```

### Transfer Files

Log into lab computer using you student email and password. (Windows)

Clone project from GitHub onto the lab computer or transfer it directly from your laptop using USB.

On the Windows lab computer, open the project folder `/CS4207-Coding-Challenge/`.

Run [`rockhill.csis.ul.ie.bat`](/rockhill.csis.ul.ie.bat) from within the folder to copy folders/files/results between lab computer and server.

**NOTE** - folders/files to copy are listed in the script as `UPLOAD_FILES=` and `DOWNLOAD_FILES=`

