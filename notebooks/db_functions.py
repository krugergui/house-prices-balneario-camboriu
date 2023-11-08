from keys import private
import sqlalchemy as SQLAlchemy
import mysql.connector

# Connecting and initiating curor
def connect_to_database_sqlalchemy() -> SQLAlchemy.engine:
    engine = SQLAlchemy.create_engine(f'mysql+pymysql://root:{private.DB_PASSWORD}@localhost:3306/imoveis-balneario-camboriu')

    return engine

def connect_to_database_mysql_connector() -> tuple:
    """Connect to DB

    Returns: 
        - db - database connection
        - cursor - database cursor
    """
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd=private.DB_PASSWORD,
        database='imoveis-balneario-camboriu'
    )

    return db, db.cursor(buffered=True)

def connect_to_test_database_mysql_connector() -> tuple:
    """Connect to DB

    Returns: 
        - db - database connection
        - cursor - database cursor
    """
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd=private.DB_PASSWORD,
        database='imoveis-balneario-camboriu-test'
    )

    return db, db.cursor(buffered=True)