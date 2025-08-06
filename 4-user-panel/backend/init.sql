CREATE DATABASE activity_vectors;
USE activity_vectors;

CREATE TABLE Activities (
    activity_id SERIAL PRIMARY KEY,
    activity_timestamp TIMESTAMP NOT NULL,
    predicted VARCHAR(50) NOT NULL CHECK (
        predicted IN (
            'backdoor', 'ddos', 'dos', 'injection', 'mitm',
            'normal', 'password', 'ransomware', 'scanning', 'xss'
        )
    )
);

CREATE TABLE Vectors (
    vector_id SERIAL PRIMARY KEY,
    activity_id INTEGER NOT NULL,
    vector_type VARCHAR(20) NOT NULL CHECK (
        vector_type IN ('dev', 'lin', 'net', 'proba')
    ),
    FOREIGN KEY (activity_id) REFERENCES Activities(activity_id)
);

CREATE TABLE VectorFeatures (
    feature_id SERIAL PRIMARY KEY,
    vector_id INTEGER NOT NULL,
    label VARCHAR(50) NOT NULL CHECK (
        label IN (
            'backdoor', 'ddos', 'dos', 'injection', 'mitm',
            'normal', 'password', 'ransomware', 'scanning', 'xss'
        )
    ),
    probability FLOAT NOT NULL,
    FOREIGN KEY (vector_id) REFERENCES Vectors(vector_id)
);
