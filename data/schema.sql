DROP TABLE IF EXISTS energy_consumption;
DROP TABLE IF EXISTS households;

CREATE TABLE households (
    household_id INT PRIMARY KEY,
    num_residents INT NOT NULL,
    house_type VARCHAR(20) NOT NULL,
    income_level VARCHAR(10) NOT NULL,
    activity_pattern INT NOT NULL,
    num_residents_original INT
);

CREATE TABLE energy_consumption (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    consumption_kwh FLOAT NOT NULL,
    temperature_c FLOAT,
    humidity_percent FLOAT,
    household_id INT,
    FOREIGN KEY (household_id) REFERENCES households(household_id)
);