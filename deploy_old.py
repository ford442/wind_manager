import os
import paramiko
import getpass

# --- Server Configuration ---
# Credentials must come from the environment or an interactive prompt — never commit secrets.
HOSTNAME = os.environ.get('DEPLOY_HOST', '1ink.us')
PORT = int(os.environ.get('DEPLOY_PORT', '22'))
USERNAME = os.environ.get('DEPLOY_USER', 'ford442')

# --- Project Configuration ---
LOCAL_DIRECTORY = 'dist'
REMOTE_DIRECTORY = os.environ.get('DEPLOY_REMOTE_DIR', 'test.1ink.us/wind')

def upload_directory(sftp_client, local_path, remote_path):
    """Recursively uploads a directory and its contents to the remote server."""
    print(f'Creating remote directory: {remote_path}')
    try:
        sftp_client.mkdir(remote_path)
    except IOError:
        print(f'Directory {remote_path} already exists.')

    for item in os.listdir(local_path):
        local_item_path = os.path.join(local_path, item)
        remote_item_path = f'{remote_path}/{item}'

        if os.path.isfile(local_item_path):
            print(f'Uploading file: {local_item_path} -> {remote_item_path}')
            sftp_client.put(local_item_path, remote_item_path)
        elif os.path.isdir(local_item_path):
            upload_directory(sftp_client, local_item_path, remote_item_path)

def main():
    """Connect to the server and upload the build output."""
    password = os.environ.get('DEPLOY_SSH_PASSWORD')
    if not password:
        password = getpass.getpass(f'Enter password for {USERNAME}@{HOSTNAME}: ')

    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((HOSTNAME, PORT))
        print('Connecting to server...')
        transport.connect(username=USERNAME, password=password)
        print('Connection successful!')

        sftp = paramiko.SFTPClient.from_transport(transport)
        print(f"Starting upload of '{LOCAL_DIRECTORY}' to '{REMOTE_DIRECTORY}'...")

        upload_directory(sftp, LOCAL_DIRECTORY, REMOTE_DIRECTORY)

        print('\n✅ Deployment complete!')

    except (paramiko.SSHException, OSError, IOError) as e:
        print(f'❌ An error occurred: {e}')
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()
        print('Connection closed.')

if __name__ == '__main__':
    if not os.path.exists(LOCAL_DIRECTORY):
        print(f"Error: Local directory '{LOCAL_DIRECTORY}' not found. Did you run 'npm run build' first?")
    else:
        main()
