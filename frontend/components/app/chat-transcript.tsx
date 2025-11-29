'use client';

import { AnimatePresence, type HTMLMotionProps, motion } from 'motion/react';
import { type ReceivedChatMessage } from '@livekit/components-react';
import { ChatEntry } from '@/components/livekit/chat-entry';

const MotionContainer = motion.create('div');
const MotionChatEntry = motion.create(ChatEntry);

const CONTAINER_MOTION_PROPS = {
  variants: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

const MESSAGE_MOTION_PROPS = {
  variants: {
    hidden: { opacity: 0, translateY: 10 },
    visible: { opacity: 1, translateY: 0 },
  },
};

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedChatMessage[];
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  ...props
}: ChatTranscriptProps & Omit<HTMLMotionProps<'div'>, 'ref'>) {
  return (
    <AnimatePresence>
      {!hidden && (
        <MotionContainer {...CONTAINER_MOTION_PROPS} {...props}>
          {messages.map(({ id, timestamp, from, message }: ReceivedChatMessage) => {
            const messageOrigin = from?.isLocal ? 'local' : 'remote';

            return (
              <div
                key={id}
                className={
                  messageOrigin === 'local'
                    ? 'bg-blue-600/30 border border-blue-500 p-3 rounded-xl text-blue-200 max-w-lg ml-auto shadow-md'
                    : 'bg-amber-100/50 border border-amber-300 p-4 rounded-xl text-amber-900 max-w-lg mr-auto shadow-md font-serif'
                }
              >
                <MotionChatEntry
                  message={message}   // ⭐ FIXED HERE
                  timestamp={timestamp}
                  messageOrigin={messageOrigin}
                  {...MESSAGE_MOTION_PROPS}
                />
              </div>
            );
          })}
        </MotionContainer>
      )}
    </AnimatePresence>
  );
}
