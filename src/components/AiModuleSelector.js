import React from 'react';

function AiModuleSelector() {
  const aiModules = [
    { id: 1, name: 'Phát Hiện Vật Thể', description: 'Mô-đun để phát hiện các vật thể trong video.' },
    { id: 2, name: 'Phân Tích Hành Vi', description: 'Phân tích các hành vi khác thường.' },
  ];

  return (
    <div className="ai-module-selector">
      <h2>Chọn Mô-đun AI</h2>
      <ul>
        {aiModules.map((module) => (
          <li key={module.id}>
            <h3>{module.name}</h3>
            <p>{module.description}</p>
            <button>Chạy Mô-đun</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default AiModuleSelector;
