import boto3
from botocore.client import Config

# Настройки подключения к MinIO
minio_config = {
    'endpoint_url': 'http://localhost:9000',
    'aws_access_key_id': 'minioadmin',
    'aws_secret_access_key': 'minioadmin',
    'config': Config(signature_version='s3v4'),
    'region_name': 'us-east-1'
}

def setup_minio():
    try:
        # Создаем клиент
        s3_client = boto3.client('s3', **minio_config)
        
        # Проверяем подключение (листинг бакетов)
        buckets = s3_client.list_buckets()
        print(" Успешное подключение к MinIO!")
        print("Существующие бакеты:", [b['Name'] for b in buckets['Buckets']])
        
        # Создаем бакет для моделей, если его нет
        bucket_name = 'ml-models'
        existing_buckets = [b['Name'] for b in buckets['Buckets']]
        
        if bucket_name not in existing_buckets:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f" Бакет '{bucket_name}' создан")
        else:
            print(f" Бакет '{bucket_name}' уже существует")
            
        return s3_client
        
    except Exception as e:
        print(f" Ошибка при подключении к MinIO: {e}")
        print("\nПроверьте:")
        print("1. Запущен ли контейнер: docker ps")
        print("2. Доступен ли MinIO: http://localhost:9001")
        return None

if __name__ == "__main__":
    client = setup_minio()