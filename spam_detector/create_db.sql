DROP TABLE IF EXISTS messages;

CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  label INTEGER NOT NULL  -- 1=spam, 0=ham
);

INSERT INTO messages (text, label) VALUES
  ('Виграй айфон! Перейди за посиланням і отримай приз', 1),
  ('Привіт! Ти завтра будеш на парі?', 0),
  ('Знижка -90% тільки сьогодні! Клікни тут', 1),
  ('Нагадування про зустріч о 14:00', 0),
  ('Терміново! Ваш рахунок заблоковано, підтвердьте дані', 1),
  ('Мама дзвонила, передзвони, будь ласка', 0),
  ('Лише сьогодні кредит без відсотків!!!', 1),
  ('Доброго ранку! Як справи?', 0);
