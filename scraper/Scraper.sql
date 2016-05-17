CREATE DATABASE `scraper` /*!40100 DEFAULT CHARACTER SET latin1 */;

CREATE TABLE `outcode` (
  `id` int(11) NOT NULL,
  `completed` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

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
