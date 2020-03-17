
import asyncio
import logging
from string import Template
from typing import Tuple, List, Any, NoReturn, Iterable, Optional

import asyncpg
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import ResultProxy

from clir.utils.singleton import Singleton


def conn_str(host: str='localhost', port:str = '5432', database: str='wiki', username: str='holocleanuser', password: str='abcd1234') -> str:
    """
    Generate a PostgreSQL connection string from connection parameters.

    :param host: the database host
    :param database: the database to connect to
    :param username: the database username
    :param password: the database password
    :return: the database connection string
    """
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"


logger = logging.getLogger(__name__)

# Template SQL statements.
index_template = Template('CREATE INDEX ON $table_name ($attr)')
drop_table_template = Template("DROP TABLE IF EXISTS $table_name CASCADE")
drop_view_template = Template("DROP VIEW IF EXISTS $view_name CASCADE")
drop_schema_template = Template("DROP SCHEMA IF EXISTS $schema_name CASCADE")
create_table_template = Template("CREATE TABLE $table_name AS ($stmt)")
table_columns_template = Template("SELECT column_name "
                                  "FROM INFORMATION_SCHEMA.COLUMNS "
                                  "WHERE table_schema = '$table_schema'"
                                  "  AND table_name = '$table_name'")
add_primary_key_template = Template("ALTER TABLE $table_name ADD PRIMARY KEY ($attr);")


def _execute_query(args: Tuple[int, str, bool], db_url: str):
    """
    Creates a connection to the database and executes a sql query.

    :param args: (query_id, query_string, fetch_results) tuple
    :param db_url: database connection string
    :return: result set
    """
    query_id = args[0]
    query_string = args[1]
    fetch_results = args[2] if len(args) > 2 else False
    logger.debug(f"Starting to execute query id {query_id}: {query_string}")
    con = psycopg2.connect(db_url)
    cur = con.cursor()
    res = cur.execute(query_string)
    if fetch_results:
        res = cur.fetchall()
    else:
        con.commit()
    con.close()
    return res


def _execute_query_w_backup(args: Tuple[int, Tuple[str, str]], db_url: str, timeout: int):
    """
    Creates a connection to the database and executes a sql query. Tries to execute the first query and if it fails,
    it sets the timeout to `timeout` then executes a backup query.

    :param args: (query_id, (query_string, query_backup_string)) tuple
    :param db_url: database connection string
    :param timeout: timeout to cancel the first query_string
    :return: result set
    """
    query_id = args[0]
    query_string = args[1][0]
    query_backup = args[1][1]
    logger.debug(f"Starting to execute query with id: {query_id}:  {query_string}")
    con = psycopg2.connect(db_url)
    cur = con.cursor()
    cur.execute(f"SET statement_timeout to {timeout}")
    try:
        cur.execute(query_string)
        res = cur.fetchall()
    except psycopg2.OperationalError:
        # TODO: change exception to psycopg2.errors.QueryCanceled error when we update psycopg2 to version 2.8.
        logger.debug(f"Query {query_id} timed out. Query: {query_string}")
        logger.debug(f"Starting to execute backup query: {query_backup}")
        con.close()
        con = psycopg2.connect(db_url)
        cur = con.cursor()
        cur.execute(query_backup)
        res = cur.fetchall()
        con.close()
    return res


class DB(metaclass=Singleton):
    """
    Communicates with a DBMS backend and performs SQL operations.
    """

    def __init__(self, db_url: str = conn_str(), pool_size=20, timeout=60000, pool_size_min: int = 10, pool_size_max: int = 20):
        """
        The constructor creates connection pools to the backend.

        :param db_url: the url of the database to connect to.
        :param pool_size: number of connections to use in psycopg2.
        :param timeout: timeout to cancel primary queries and fall back to secondary queries.
        :param pool_size_min: initial number of connections to initialize the database pool in asyncpg.
        :param pool_size_max: max number of connections in asyncpg.
        """
        self.db_url = db_url
        self.timeout = timeout
        self.pool_size = pool_size
        self.engine = create_engine(db_url, client_encoding='utf8', pool_size=pool_size)

        async def create_async_pool():
            return await asyncpg.create_pool(dsn=db_url, min_size=pool_size_min, max_size=pool_size_max)

        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        self.async_pool = self.event_loop.run_until_complete(create_async_pool())

    def export_query_to_csv(self, query: str, path: str, size: int = 8192) -> NoReturn:
        """
        Writes the output of a SQL query to a CSV file.

        :param query: SQL query.
        :param path: file path to write to.
        :param size: buffer size.
        """
        conn = self.engine.connect()
        cur = conn.connection.cursor()
        copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
        with open(path, "w") as file:
            cur.copy_expert(sql=copy_sql, file=file, size=size)
        conn.close()

    def execute_queries(self, queries: List[str]) -> List:
        """
        Serially executes a list of queries.

        :param queries: List of SQL query strings.
        :return: list of results for each query, matching the input queries order.
        """
        logger.debug(f"Preparing to execute {len(queries)} queries.")
        results = []
        for idx, q in enumerate(queries):
            results.append(_execute_query((idx, q, True), self.db_url))
        return results

    def execute_queries_w_backup(self, queries: List[str]) -> List:
        """
        Executes queries that have backups in parallel. Used in featurization.

        :param queries: list of queries to execute
        :return: query execution results.
        """
        logger.debug(f"Preparing to execute {len(queries)} queries.")
        results = []
        for idx, q in enumerate(queries):
            results.append(_execute_query_w_backup((idx, q), self.db_url, self.timeout))
        return results

    def execute_query(self, query: str):
        """
        Executes a single SQL query and returns the results.

        :param query: query string
        :return: the result set of the query
        """
        conn = self.engine.connect()
        result = conn.execute(query).fetchall()
        conn.close()
        return result

    def execute_get_cursor(self, query: str) -> ResultProxy:
        """
        Connectionless execution of query to auto-close connection after the cursor
        is closed.

        :param query: SQL query to execute.
        :return: a proxy for the results to be consumed on-demand.
        """
        cursor = self.engine.execute(query)  # Connectionless operation to auto-close connection.
        return cursor

    def execute_queries_parallel(self, queries: List[str]) -> List[asyncpg.Record]:
        """
        Uses the self.async_pool to execute queries in parallel. The queries MUST return the same
        schema because the results of all queries are merged together in one list.

        :param queries: list of queries to execute.
        :return: one list of results, each element is a Record that can be accessed using its keys.
        """

        async def execute_single_query(q: str) -> List[asyncpg.Record]:
            return await self.async_pool.fetch(q)

        asyncio.set_event_loop(self.event_loop)
        all_calls = asyncio.gather(*[execute_single_query(q) for q in queries])
        results = self.event_loop.run_until_complete(all_calls)
        all_results = []
        for result in results:
            all_results.extend(result)
        return all_results

    def execute_updates_parallel(self, queries: List[str]) -> NoReturn:
        """
        Executes a list of update queries in parallel.

        :param queries: list of SQL update queries.
        """

        async def execute_single_update(q: str):
            await self.async_pool.execute(q)

        asyncio.set_event_loop(self.event_loop)
        all_calls = asyncio.gather(*[execute_single_update(q) for q in queries])
        self.event_loop.run_until_complete(all_calls)

    def insert_records_parallel(self, schema_name: str, table_name: str, records: Iterable, columns: Optional[List[str]] = None) -> NoReturn:
        """
        Inserts a list of records in parallel into the database using asyncpg.
        Significantly faster than inserting batches or inserting one tuple at a time.

        :param schema_name: name of schema.
        :param table_name: table name to insert into.
        :param records: an iterable collection of records to insert into the table.
        :param columns: optional list of columns to insert the (partial) records into if the
          records do not fill the whole table schema.
        """

        async def run():
            async with self.async_pool.acquire() as con:
                await con.copy_records_to_table(schema_name=schema_name, table_name=table_name,
                                                columns=columns, records=records)

        self.event_loop.run_until_complete(run())

    def copy_file_to_table(self,
                           schema_name: str,
                           table_name: str,
                           attributes: List[str],
                           file: Any,
                           delimiter: str = None) -> NoReturn:
        """
        Loads a file directly into a database table without going through python in-memory operations.
        Assumes that the file is in CSV format. Uses asyncpg driver.

        :param schema_name: schema name.
        :param table_name: table to copy the file to.
        :param attributes: list of attributes to copy.
        :param file: file handle or path.
        :param delimiter: what delimiter to use.
        """

        async def run():
            async with self.async_pool.acquire() as con:
                await con.copy_to_table(
                    schema_name=schema_name,
                    table_name=table_name,
                    columns=attributes,
                    source=file,
                    delimiter=delimiter,
                    header=True,
                    format='csv',
                )

        self.event_loop.run_until_complete(run())

    def copy_table_to_file(self,
                           schema_name: str,
                           table_name: str,
                           attributes: List[str],
                           file_path: str,
                           delimiter: str = None) -> NoReturn:
        """
        Dumps a database table to a CSV file.

        :param schema_name: schema name.
        :param table_name: the name of the table to dump into the file.
        :param attributes: list of attributes to dump.
        :param file_path: the path to which we copy the table to.
        :param delimiter: what delimiter to use as a separator.
        """

        async def run():
            async with self.async_pool.acquire() as con:
                await con.copy_from_table(
                    schema_name=schema_name,
                    table_name=table_name,
                    columns=attributes,
                    output=file_path,
                    delimiter=delimiter,
                    header=True,
                    format='csv',
                )

        self.event_loop.run_until_complete(run())

    def execute_update(self, query: str) -> NoReturn:
        """
        Executes an update query without returning results.

        :param query: SQL query.
        """
        conn = self.engine.connect()
        ret = conn.execute(query)
        conn.close()

    def execute_updates(self, queries: List[str]) -> NoReturn:
        """
        Executes multiple update SQL statements without returning results. NOTE: Order is not guaranteed since they
        execute in parallel.

        :param queries: list of queries to execute
        """
        logger.debug(f"Preparing to execute {len(queries)} queries.")
        results = []
        for idx, q in enumerate(queries):
            results.append(_execute_query((idx, q, False), self.db_url))

    def create_db_table_from_query(self, table_name: str, query_str: str) -> bool:
        """
        Creates a database table from a SQl query and populates it with the output of the query.
        If a table with the same name exists, we drop it first.

        :param table_name: the name of the table to be created
        :param query_str: the sql query string that populates the table
        """
        drop = drop_table_template.substitute(table_name=table_name)
        create = create_table_template.substitute(table_name=table_name, stmt=query_str)
        conn = self.engine.connect()
        dropped = conn.execute(drop)
        created = conn.execute(create)
        conn.close()
        return True

    def drop_table(self, table_name: str) -> NoReturn:
        """
        Drops a table if it exists.
        """
        drop = drop_table_template.substitute(table_name=table_name)
        conn = self.engine.connect()
        conn.execute(drop)
        conn.close()

    def drop_view(self, view_name: str) -> NoReturn:
        """
        Drops a view if it exists.
        """
        drop = drop_view_template.substitute(view_name=view_name)
        conn = self.engine.connect()
        conn.execute(drop)
        conn.close()

    def drop_schema(self, schema_name: str) -> NoReturn:
        """
        Drops a schema if it exists.
        """
        drop = drop_schema_template.substitute(schema_name=schema_name)
        conn = self.engine.connect()
        conn.execute(drop)
        conn.close()

    def create_db_index(self, table_name: str, attr_list: List[str]) -> Any:
        """
        Creates an index on a list of attributes in a table.

        :param table_name: the table name
        :param attr_list: the list of attributes to index
        :return: the result of creating the index
        """
        stmt = index_template.substitute(table_name=table_name, attr=','.join(attr_list))
        conn = self.engine.connect()
        result = conn.execute(stmt)
        conn.close()
        return result

    def add_table_pk(self, table_name: str, attr: str) -> Any:
        """
        Add a primary key to a table.

        :param table_name: the table name
        :param attr: the attributes for the pk
        :return: the result of creating the index
        """
        stmt = add_primary_key_template.substitute(table_name=table_name, attr=attr)
        conn = self.engine.connect()
        result = conn.execute(stmt)
        conn.close()
        return result

    def get_table_columns(self, schema_name: str, table_name: str) -> List[str]:
        """
        Retrieves the columns of a given table.

        :param schema_name: schema name
        :param table_name: name of the table
        :return: list of columns defined in this table in the database
        """
        stmt = table_columns_template.substitute(table_schema=schema_name, table_name=table_name)
        conn = self.engine.connect()
        result = conn.execute(stmt).fetchall()
        conn.close()
        # Extract column names from the tuples
        column_names = [row[0] for row in result]
        return column_names

    def table_exists(self, schema_name: str, table_name: str) -> bool:
        """
        Checks whether or not a table exists.

        :param schema_name: schema name.
        :param table_name: table name.
        :return: True if the table exists in the specified schema, False otherwise.
        """
        sql = f"""
            SELECT EXISTS (
                SELECT 1
                FROM   information_schema.tables 
                WHERE  table_schema = '{schema_name}'
                  AND  table_name = '{table_name}'
            )
        """
        result = self.execute_query(sql)
        return result[0][0]

    def view_exists(self, schema_name: str, view_name: str) -> bool:
        """
        Checks whether or not a view exists.

        :param schema_name: schema name.
        :param view_name: view name.
        :return: True if the view exists in the specified schema, False otherwise.
        """
        sql = f"""
            SELECT EXISTS (
                SELECT 1
                FROM   information_schema.views
                WHERE  table_schema = '{schema_name}'
                  AND  table_name = '{view_name}'
            )
        """
        result = self.execute_query(sql)
        return result[0][0]

    def sequence_exists(self, schema_name: str, sequence_name: str) -> bool:
        """
        Checks whether or not a sequence exists.

        :param schema_name: schema name.
        :param sequence_name: sequence name.
        :return: True if the sequence exists in the specified schema, False otherwise.
        """
        sql = f"""
            SELECT EXISTS (
                SELECT 1
                FROM   information_schema.sequences
                WHERE  sequence_schema = '{schema_name}'
                  AND  sequence_name = '{sequence_name}'
            )
        """
        result = self.execute_query(sql)
        return result[0][0]

    def schema_exists(self, schema_name: str) -> bool:
        """
        Checks whether or not a schema exists in the database

        :param schema_name: schema name.
        :return: True if the schema exists in the database, False otherwise.
        """
        sql = f"""
        SELECT EXISTS (
            SELECT 1
            FROM  information_schema.schemata
            WHERE schema_name = '{schema_name}'
        )
        """
        result = self.execute_query(sql)
        return result[0][0]
