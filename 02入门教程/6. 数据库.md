# 1. 数据库介绍
数据库是用于存储、管理和操作大量结构化数据的系统。

在数据库中，数据以表格的形式进行组织，每个表格包含多个行和列，分别代表不同的数据记录和属性。通过使用数据库管理系统（DBMS），我们可以创建、修改和查询数据库中的数据，实现数据的增删改查操作。

# 2. 技术路线

## 2.1 数据库设计
数据库设计是构建数据库的重要步骤，它涉及确定数据库的结构、关系和约束条件。

## 2.2 sql语句
SQL（Structured Query Language）是一种用于与数据库进行交互的标准化语言。通过使用SQL语句，我们可以执行各种数据库操作，包括创建表格、插入数据、查询数据、更新数据和删除数据等。

1. 创建表格
```
CREATE TABLE students (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    gender VARCHAR(10)
);
```
2. 插入数据
```
INSERT INTO students (id, name, age, gender) VALUES (1, 'Alice', 20, 'Female');
INSERT INTO students (id, name, age, gender) VALUES (2, 'Bob', 22, 'Male');
```

3. 查询数据
```
SELECT * FROM students;
SELECT name, age FROM students WHERE gender = 'Female';
```

4. 更新数据
```
UPDATE students SET age = 21 WHERE id = 1;
```

5. 删除数据
```
DELETE FROM students WHERE id = 2;
```


## 2.3 SQLite工具

SQLite是一种轻量级的嵌入式数据库引擎，它提供了简单易用的数据库管理功能。与传统的客户-服务器模式的数据库系统不同，SQLite以文件形式存储数据库，可以直接嵌入到应用程序中，无需独立的数据库服务器。
