import Providers from './providers';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'AI 퀴즈 생성기',
  description: 'PDF 문서를 기반으로 AI가 자동으로 문제를 생성하고 학습을 도와주는 시스템',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}