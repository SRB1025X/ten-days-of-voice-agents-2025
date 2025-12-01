'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import type { AppConfig } from '@/app-config';
import { ChatTranscript } from '@/components/app/chat-transcript';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import { TileLayout } from '@/components/app/tile-layout';
import { AgentControlBar, type ControlBarControls } from '@/components/livekit/agent-control-bar/agent-control-bar';
import { useChatMessages } from '@/hooks/useChatMessages';
import { useConnectionTimeout } from '@/hooks/useConnectionTimout';
import { useDebugMode } from '@/hooks/useDebug';
import { cn } from '@/lib/utils';
import { ScrollArea } from '../livekit/scroll-area/scroll-area';
import { Button } from '@/components/livekit/button';

const MotionBottom = motion.create('div');
const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

const BOTTOM_VIEW_MOTION_PROPS = {
  variants: {
    visible: { opacity: 1, translateY: '0%' },
    hidden: { opacity: 0, translateY: '100%' },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: { duration: 0.3, delay: 0.5, ease: 'easeOut' },
};

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & { appConfig: AppConfig }) => {
  useConnectionTimeout(200_000);
  useDebugMode({ enabled: IN_DEVELOPMENT });

  const messages = useChatMessages();
  const [chatOpen, setChatOpen] = useState(true); // ALWAYS show chat for game
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: false,
    camera: false,
    screenShare: false,
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <section className="bg-background relative z-10 h-full w-full overflow-hidden" {...props}>
      <div className="fixed inset-0 grid grid-cols-1 grid-rows-1">
        <ScrollArea ref={scrollAreaRef} className="px-4 pt-24 pb-[150px]">
          <ChatTranscript
            hidden={false}
            messages={messages}
            className="mx-auto max-w-2xl space-y-3"
          />
        </ScrollArea>
      </div>

      <TileLayout chatOpen={chatOpen} />

      <MotionBottom
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="fixed inset-x-3 bottom-0 z-50 md:inset-x-12"
      >
        <div className="bg-background relative mx-auto max-w-2xl pb-6">
          <AgentControlBar controls={controls} onChatOpenChange={setChatOpen} />

          {/* ⭐ Restart Improv Battle */}
          <div className="w-full flex justify-center mt-3">
            <Button
              variant="primary"
              size="md"
              className="w-40 font-mono"
              onClick={() => window.location.reload()}
            >
              Restart Battle
            </Button>
          </div>
        </div>
      </MotionBottom>
    </section>
  );
};
