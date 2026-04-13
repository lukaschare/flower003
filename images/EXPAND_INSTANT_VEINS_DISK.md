
# Expanding the Instant Veins VM Disk to 80 GB

FedITS-Tool requires substantially more disk space than the default Instant Veins VM allocation.
Before running the setup script, expand the VM disk/partition to at least 80 GB.


## Step 0. Expand the default Instant Veins VM disk from 20 GB to 80 GB

The default Instant Veins VirtualBox image typically provides only a 20 GB system disk, which is insufficient for FedITS-Tool.    
Before installation, manually increase the virtual disk size to **80 GB** in VirtualBox.

![[Pasted image 20260413155546.png]]
## Step 1: Install the partition growth utility

```bash
sudo apt update
sudo apt install -y cloud-guest-utils
````

## Step 2: Expand partition `/dev/sda1`

```bash
sudo growpart /dev/sda 1
```

If successful, you should see output similar to:

```text
CHANGED: partition=1 start=...
```

## Step 3: Resize the filesystem

```bash
sudo resize2fs /dev/sda1
```

## Step 4: Verify the new disk size

```bash
df -h /
lsblk
```

```
Example:

veins:~% df -h /
lsblk
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        79G   23G   53G  30% /
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sda      8:0    0   80G  0 disk 
└─sda1   8:1    0   80G  0 part /
sr0     11:0    1 52.4M  0 rom  
```