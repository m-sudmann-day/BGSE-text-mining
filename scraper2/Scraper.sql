CREATE DATABASE `scraper` /*!40100 DEFAULT CHARACTER SET latin1 */;
USE `scraper`;

-- Create a table storing the outcodes scraped so far and a binary 
-- indicating whether the scraping was succesfully completed.

CREATE TABLE `outcode` (
  `id` int(11) NOT NULL,
  `completed` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Create main fact table storing the description and metadata for 
-- each property.

CREATE TABLE `property` (
  `id` int(11) NOT NULL,
  `outcode_id` int(11) NOT NULL,
  `url` varchar(250) NOT NULL,
  `title` varchar(250) DEFAULT NULL,
  `description` text,
  `monthly_price` int(11) DEFAULT NULL,
  `weekly_price` int(11) DEFAULT NULL,
  `latitude` float DEFAULT NULL,
  `longitude` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `outcode_id` (`outcode_id`),
  CONSTRAINT `property_ibfk_1` FOREIGN KEY (`outcode_id`) REFERENCES `outcode` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Create a table storing the errors thrown during the process
-- the object type that produced them and the error message.
CREATE TABLE `error` (
  `obj_type` varchar(250) DEFAULT NULL,
  `id` varchar(20) DEFAULT NULL,
  `text` text
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
