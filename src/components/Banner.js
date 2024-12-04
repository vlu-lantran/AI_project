import React from 'react';
import '../App.css'; // Đường dẫn đến file CSS đã tạo ở trên

const Banner = ({ imageUrl, altText }) => {
  return (
    <div className="banner">
      <img src={imageUrl} alt={altText} />
    </div>
  );
};

export default Banner;
